//===-- HelloWorld.cpp - Example Transformations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/HelloWorld.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <fstream>
#include <iostream>

// asan--
#include "SlimasanProject.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/CFG.h"

using namespace llvm;

uint64_t safecnt = 0;

bool HelloWorldPass::isNoInstrument(Value *V) {
    if (V && V->hasName()) {
        StringRef Name = V->getName();
        if (Name.startswith(NOINSTRUMENT_PREFIX))
            return true;
        // Support for mangled C++ names (should maybe do something smarter here)
        if (Name.startswith("_Z"))
            return Name.find(NOINSTRUMENT_PREFIX, 2) != StringRef::npos;
    }
    return false;
}

bool HelloWorldPass::shouldInstrument(Function &F) {
  if (F.isDeclaration())
      return false;

  if (isNoInstrument(&F))
      return false;

  // openmp
  if(F.getName().startswith(".omp"))
      return false;

  // disable sanitizer attribute
  if(F.hasFnAttribute(llvm::Attribute::DisableSanitizerInstrumentation))
      return false;

  // safestack runtime
  if(F.getName() == "__safestack_init")
      return false;

  return true;
}

void HelloWorldPass::getInterestingMemoryOperands(Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  if (I->hasMetadata("nosanitize") || I->hasMetadata("swiftsan")) {
    return;
  }

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                                      LI->getType(), LI->getAlign());
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                                      SI->getValueOperand()->getType(), SI->getAlign());
  }
}

void markAsNonCheck(Function &F, Instruction *I) {
  // set metadata
  // errs() << "Marking as safe aka NO CHECK: " << *I << "\n";
  I->setMetadata(F.getParent()->getMDKindID("slimasan"), llvm::MDNode::get(F.getContext(), std::nullopt));
  safecnt++;
}

void ConservativeCallIntrinsicCollect(Function &F, std::set<Instruction *> &callIntrinsicSet) {

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      // Here we check if current instruction is call instruction
      if (dyn_cast<CallInst>(&Inst)) {
        callIntrinsicSet.insert(&Inst);
        continue;
      }
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&Inst);
      // Here we check if Intrinsic ID is lifetime_end
      if (II && II->getIntrinsicID() == Intrinsic::lifetime_end) {
        callIntrinsicSet.insert(&Inst);
        continue;
      }
    }
  }
}

bool isPostDominatWrapper(Instruction *InstStart, Instruction *TargetInst, llvm::PostDominatorTree &PDT) {
  
  BasicBlock *StartBB = InstStart->getParent();
  BasicBlock *TargetBB = TargetInst->getParent();
  if (StartBB == TargetBB) {
    for (auto &itrInst : *StartBB) {
      if (&itrInst == InstStart) {
        return false;
      }
      if (&itrInst == TargetInst) {
        return true;
      }
    }
  }
  return PDT.dominates(StartBB, TargetBB);
}

bool ConservativeCallIntrinsicCheck(Instruction *InstStart, Instruction *InstEnd, std::set<Instruction *> &callIntrinsicSet, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT) {

  for (auto TargetInst : callIntrinsicSet) {
    // InstStart -> TargetInst -> InstEnd && InstStart !PostDominat TargetInst
    if (isPotentiallyReachable(InstStart, TargetInst) && isPotentiallyReachable(TargetInst, InstEnd) && !isPostDominatWrapper(InstStart, TargetInst, PDT)) {
      return false;
    }
  }
  return true;
}

void sequentialExecuteOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA) {
  auto DT = DominatorTree(F);
  std::map<Value *, std::set<InterestingMemoryOperand *>> AddrToInstructions;
  auto PDT = PostDominatorTree();
  PDT.recalculate(F);

  // pre-processing
  // group instructions that access the same address (alias considered)
  for (InterestingMemoryOperand &Operand : OperandsToInstrument) {
    if (Value *Addr = Operand.getPtr()) {

      if (AddrToInstructions.find(Addr) == AddrToInstructions.end()) {

        bool aliasFound = false;
        //handle the possibility of alias
        for (auto item : AddrToInstructions) {
          if (AA->isMustAlias(item.first, Addr)) {
            aliasFound = true;
            AddrToInstructions[item.first].insert(&Operand);
            break;
          }
        }
        //found an alias, done
        if (aliasFound) continue;

        //never appeared in the map, so add a slot
        AddrToInstructions.insert(std::pair<Value *, std::set<InterestingMemoryOperand *>>(Addr, std::set<InterestingMemoryOperand *>()));
      }
      //add the inst to the target slot (either the newly created one or an existing one)
      AddrToInstructions[Addr].insert(&Operand);
    }
  }

  std::set<Instruction *> deleted;

  std::set<Instruction *> callIntrinsicSet; 

  ConservativeCallIntrinsicCollect(F, callIntrinsicSet);

  for (auto item : AddrToInstructions) {
    for (auto inst1 : item.second) {
      //well, the instruction has been deleted, so who cares
      if (deleted.find(inst1->getInsn()) != deleted.end())
        continue;

      for (auto inst2 : item.second) {
        //avoid checking itself
        if (inst1->getInsn() == inst2->getInsn() || deleted.find(inst2->getInsn()) != deleted.end())
          continue;	

        if (DT.dominates(inst1->getInsn(), inst2->getInsn()) && ConservativeCallIntrinsicCheck(inst1->getInsn(), inst2->getInsn(), callIntrinsicSet, DT, PDT)){
          deleted.insert(inst2->getInsn());
        }
      }
    }
  }
  //Let's only keep the non-deleted ones
  SmallVector<InterestingMemoryOperand, 16> SEOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  for (auto item: SEOTempToInstrument) {
    if (deleted.find(item.getInsn()) == deleted.end())
      OperandsToInstrument.push_back(item);
    else
      markAsNonCheck(F, item.getInsn());
  }
}

void sequentialExecuteOptimizationPostDom(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA) {
  auto PDT = PostDominatorTree();
  PDT.recalculate(F);
  std::map<Value *, std::set<InterestingMemoryOperand *>> AddrToInstructions;

  // pre-processing
  // group instructions that access the same address (alias considered)
  for (InterestingMemoryOperand &Operand : OperandsToInstrument) {
    if (Value *Addr = Operand.getPtr()) {

      if (AddrToInstructions.find(Addr) == AddrToInstructions.end()) {

        bool aliasFound = false;
        //handle the possibility of alias
        for (auto item : AddrToInstructions) {
          if (AA->isMustAlias(item.first, Addr)) {
            aliasFound = true;
            AddrToInstructions[item.first].insert(&Operand);
            break;
          }
        }
        //found an alias, done
        if (aliasFound) continue;

        //never appeared in the map, so add a slot
        AddrToInstructions.insert(std::pair<Value *, std::set<InterestingMemoryOperand *>>(Addr, std::set<InterestingMemoryOperand *>()));
      }
      //add the inst to the target slot (either the newly created one or an existing one)
      AddrToInstructions[Addr].insert(&Operand);
    }
  }

  std::set<Instruction *> deleted;

  for (auto item : AddrToInstructions) {
    for (auto inst1 : item.second) {
      //well, the instruction has been deleted, so who cares
      if (deleted.find(inst1->getInsn()) != deleted.end())
        continue;

      for (auto inst2 : item.second) {
        //avoid checking itself
        if (inst1->getInsn() == inst2->getInsn() || deleted.find(inst2->getInsn()) != deleted.end())
          continue;	

        if (PDT.dominates(inst1->getInsn()->getParent(), inst2->getInsn()->getParent())){
          deleted.insert(inst2->getInsn());
        }
      }
    }
  }
  //Let's only keep the non-deleted ones
  SmallVector<InterestingMemoryOperand, 16> SEOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  for (auto item: SEOTempToInstrument) {
    if (deleted.find(item.getInsn()) == deleted.end()){
      OperandsToInstrument.push_back(item);
    }
    else{
      markAsNonCheck(F, item.getInsn());
    }
  }
}

void slimAsanOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA, LoopInfo *LI, ScalarEvolution *SE, ObjectSizeOffsetVisitor &ObjSizeVis) {
  // ASAN--: Removing Recurring Checks
  sequentialExecuteOptimizationPostDom(F, OperandsToInstrument, AA);
  sequentialExecuteOptimization(F, OperandsToInstrument, AA);
}

using OffsetDir = uint8_t;
static constexpr OffsetDir kOffsetUnknown = 0b00;
static constexpr OffsetDir kOffsetPositive = 0b01;
static constexpr OffsetDir kOffsetNegative = 0b10;
static constexpr OffsetDir kOffsetBoth = kOffsetPositive | kOffsetNegative;


void metadataReuse(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA, ScalarEvolution *SE) {
  errs() << "entering metadata reuse for " << F.getName() << "\n";
  auto DT = DominatorTree(F);

  // gather all loads/stores that have are on a GEP with positive offset
  // then look for each one, if it dominates another, we can reuse the meta
  // use alias analysis to confirm it uses the same base

  DenseMap<Instruction *, GetElementPtrInst *> AccessPosGEP;

  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {
    errs() << "Looking at Load/Store: " << *Oper.getInsn() << "\n";
    if (Value *Addr = Oper.getPtr()) {
      if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr->stripPointerCasts())){
        errs() << "Looking at GEP: " << *GEP << "\n";
        // "ShadowBound" GEP direction analysis
        OffsetDir Dir = kOffsetUnknown;
        for (auto &Op : GEP->indices()) {
          auto Range = SE->getSCEV(Op.get());
          if (SE->getSignedRangeMin(Range).isNonNegative()){
            errs() << "Range is positive: " << *Range << "\n";
            Dir |= kOffsetPositive;
          }
          else if (SE->getSignedRangeMax(Range).isNegative()){
            errs() << "Range is negative: " << *Range << "\n";
            Dir |= kOffsetNegative;
          }
          else{
            errs() << "Range is undecided: " << *Range << "\n";
            Dir |= kOffsetBoth;
          }

          if (Dir == kOffsetBoth)
            break;
        }

        if(Dir == kOffsetPositive){
          errs() << "Positive direction GEP: " << *GEP << "\n";
          AccessPosGEP.insert({Oper.getInsn(), GEP});
        }
      }
    }
  }

  // alias analysis on bases + dominator on access

  // TODO: somehow sync up the "sharing"
  // e.g. %14 can re-use %8 and %11, but %11 can reuse %8, so we want everything to use 8

  // map from load/store instructions to other load/store instructions of which the key can re-use the metadata
  DenseMap<Instruction *, std::set<Instruction *>> MetaReuseMap;

  for(auto entry : AccessPosGEP) {
    Instruction *access = entry.first;
    GetElementPtrInst *offset = entry.second;
    for (auto other : AccessPosGEP) {
      if(access == other.first) continue;

      if(!AA->isMustAlias(offset->getPointerOperand(), other.second->getPointerOperand()))
        continue;

      if(!DT.dominates(access, other.first))
        continue;

      // here: access dominates the other access and use the same base
      errs() << "Access: " << *other.first << "\n";
      errs() << "Can reuse metadata of: " << *access << "\n";

      if(MetaReuseMap.count(access)){ // key exists
        MetaReuseMap[access].insert(other.first);
      }
      else{
        std::set<Instruction *> init;
        init.insert(other.first);
        MetaReuseMap.insert({access, init});
      }
    }
  }

  errs() << "\n Meta Reuse Map: \n";
  for (auto entry : MetaReuseMap) {
    Instruction *access = entry.first;
    errs() << "Target Access: " << *access << "\n";
    std::set<Instruction *> reusables = entry.second;

    std::set<Instruction *> redundants;

    for(Instruction *reuse : reusables) {
      for(Instruction *other : reusables) {
        if (reuse == other) continue;

        if(DT.dominates(reuse, other)){
          errs() << "Redundant: " << *other << "\n";
          redundants.insert(other);
        }
      }
      errs() << "can reuse: " << *reuse << "\n";
    }

    for(Instruction *red : redundants){
      MetaReuseMap[access].erase(red);
    }
  }

  errs() << "\n FINAL Meta Reuse Map: \n";
  for (auto entry : MetaReuseMap) {
    Instruction *access = entry.first;
    errs() << "Target Access: " << *access << "\n";
    std::set<Instruction *> reusables = entry.second;
    errs() << "Reuse: " << **(reusables.begin()) << "\n";
  }

}

bool HelloWorldPass::runOnFunc(Function &F, FunctionAnalysisManager &AM) {

#if 0
  SmallPtrSet<Value *, 16> TempsToInstrument;
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;

   // Fill the set of memory operations to instrument.
  for (auto &BB : F) {
    TempsToInstrument.clear();
    for (auto &Inst : BB) {
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInterestingMemoryOperands(&Inst, InterestingOperands);

      if (!InterestingOperands.empty()) {
        for (auto &Operand : InterestingOperands) {
          Value *Ptr = Operand.getPtr();
          // If we have a mask, skip instrumentation if we've already
          // instrumented the full object. But don't add to TempsToInstrument
          // because we might get another load/store with a different mask.
          if (Operand.MaybeMask) {
            if (TempsToInstrument.count(Ptr))
              continue; // We've seen this (whole) temp in the current BB.
          } else {
            if (!TempsToInstrument.insert(Ptr).second)
              continue; // We've seen this temp in the current BB.
          }
          OperandsToInstrument.push_back(Operand);
        }
      }
    } 
  }


  AAResults *AA = &AM.getResult<AAManager>(F);
  TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  DominatorTree DT(F);
  AssumptionCache AC(F);
  LoopInfo LI(DT);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const DataLayout &DL = F.getParent()->getDataLayout();
  ObjectSizeOpts ObjSizeOpts;
  ObjSizeOpts.RoundToAlign = true;
  ObjectSizeOffsetVisitor ObjSizeVis(DL, &TLI, F.getContext(), ObjSizeOpts);
  // slimAsanOptimization(F, OperandsToInstrument, AA, &LI, &SE, ObjSizeVis);

  // metadataReuse(F, OperandsToInstrument, AA, &SE);

  // errs() << "--Total marked as safe from IR: " << safecnt << "\n";
#endif
  return true;
}



PreservedAnalyses HelloWorldPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  bool Changed = false;
#if 0
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (auto &F : M){
      if (shouldInstrument(F) && F.hasFnAttribute(Attribute::SafeStack)) {
      Changed |= runOnFunc(F, FAM);
    }
  }
#endif
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}