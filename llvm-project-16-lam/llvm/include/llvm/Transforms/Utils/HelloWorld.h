//===-- HelloWorld.h - Example Transformations ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_HELLOWORLD_H
#define LLVM_TRANSFORMS_UTILS_HELLOWORLD_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class InterestingMemoryOperand {
public:
  Use *PtrUse;
  bool IsWrite;
  Type *OpType;
  uint64_t TypeSize;
  MaybeAlign Alignment;
  // The mask Value, if we're looking at a masked load/store.
  Value *MaybeMask;

  InterestingMemoryOperand(Instruction *I, unsigned OperandNo, bool IsWrite,
                           class Type *OpType, MaybeAlign Alignment,
                           Value *MaybeMask = nullptr)
      : IsWrite(IsWrite), OpType(OpType), Alignment(Alignment),
        MaybeMask(MaybeMask) {
    const DataLayout &DL = I->getModule()->getDataLayout();
    TypeSize = DL.getTypeStoreSizeInBits(OpType);
    PtrUse = &I->getOperandUse(OperandNo);
  }

  Instruction *getInsn() { return cast<Instruction>(PtrUse->getUser()); }

  Value *getPtr() { return PtrUse->get(); }
};

class HelloWorldPass : public PassInfoMixin<HelloWorldPass> {
  #define NOINSTRUMENT_PREFIX "__noinstrument_"
private:
  void getInterestingMemoryOperands(Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);
  bool isNoInstrument(Value *V);
  bool shouldInstrument(Function &F);
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  bool runOnFunc(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_HELLOWORLD_H
