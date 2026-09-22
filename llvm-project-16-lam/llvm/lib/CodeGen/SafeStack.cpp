//===- SafeStack.cpp - Safe Stack Insertion -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass splits the stack into the safe stack (kept as-is for LLVM backend)
// and the unsafe stack (explicitly allocated and managed through the runtime
// support library).
//
// http://clang.llvm.org/docs/SafeStack.html
//
//===----------------------------------------------------------------------===//

#include "SafeStackLayout.h"
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
#include <time.h>

// swiftsan
#include <llvm/Analysis/DominanceFrontier.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DebugInfo.h>

// swiftsan opts
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"

//test
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"

// bzhi
#include "llvm/IR/IntrinsicsX86.h"

//asan--
#include "../../llvm/lib/Transforms/Utils/SlimasanProject.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/CFG.h"
// ASAN-- Scalable Value
#define RZ_SIZE 16

// scev range
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace llvm::safestack;

#define BASEBOUNDS_X86

#define ENABLE_RESTOREPOINT_ALL 1

// AArch64 relevant only
#define ALLOCA_TAGGING

const DenseSet<StringRef> interceptors = {
  // mem family
  "memcpy", "memset", "memmove", "memcmp",
  // str family
  "strcmp", "strncmp", "strlen", "strnlen",
  "strcat", "strncat", "strcpy", "strncpy", "strdup",
  // int parse
  "atoi", "atol", "atoll", "strtol", "strtoll",
  // stdio
  "printf", "snprintf", "puts",
  // mmap
  "mmap", "munmap",
  "malloc", "calloc", "realloc", "free",
};

const std::string interceptPrefix = "__subsan_";

// if true: implicit tagging (experimental, tag bits 41+), else: explicit
// pointer-tag scheme — AArch64 TBI or x86_64 LAM_U57.
bool isImplicitTagging;
// if false: assume AArch64
bool isX86;
// Explicit-tag ABI parameters set in runOnModule() per target triple.
//   AArch64 TBI       : shift=56, fastPathMask=0xFFFF000000000000, idxMask=0x000FFFFFFFFFFFFF
//                       tag namespaces: stack 1..85, global 86..170, heap 171..255
//   x86_64 LAM_U57    : shift=57, fastPathMask=0xFE00000000000000, idxMask=0x001FFFFFFFFFFFFF
//                       tag namespaces: stack 1..21, global 22..42, heap 43..63
uint64_t subsanTagShift = 56;
uint64_t subsanFastPathMask = 0xFFFF000000000000ULL;
uint64_t subsanGranuleIdxMask = 0x000FFFFFFFFFFFFFULL;
uint8_t  subsanStackTagLo = 1;
uint8_t  subsanStackTagHi = 85;
uint8_t  subsanGlobalTagLo = 86;
uint8_t  subsanGlobalTagHi = 170;
Type* cursedType;

using OffsetDir = uint8_t;
static constexpr OffsetDir kOffsetUnknown = 0b00;
static constexpr OffsetDir kOffsetPositive = 0b01;
static constexpr OffsetDir kOffsetNegative = 0b10;
static constexpr OffsetDir kOffsetBoth = kOffsetPositive | kOffsetNegative;


DenseMap<Instruction*, std::vector<size_t>> RestorePoints;

#define DEBUG_TYPE "safe-stack"

#define FAIL(msg) do {                                     \
    llvm::errs() << "[SizedStack] error: " << msg << "\n"; \
    exit(1);                                               \
  } while (0);


#define NOINSTRUMENT_PREFIX "__noinstrument_"
bool isNoInstrument(Value *V) {
    if (V && V->hasName()) {
        StringRef Name = V->getName();
        if (Name.startswith(NOINSTRUMENT_PREFIX))
            return true;
        if (Name.startswith("__subsan_"))
            return true;
        if (Name.startswith("__safestack_") || Name.startswith("__hwasan_"))
            return true;
        // Support for mangled C++ names (should maybe do something smarter here)
        if (Name.startswith("_Z"))
            return Name.find(NOINSTRUMENT_PREFIX, 2) != StringRef::npos;
    }
    return false;
}

void setNoInstrument(Value *V) {
    V->setName(std::string(NOINSTRUMENT_PREFIX) + V->getName().str());
}

bool shouldInstrument(Function &F) {
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

bool shouldInstrumentGlobal(GlobalVariable &GV) {
  // Ignore globals from runtime instrumentation runtime.
  if (isNoInstrument(&GV) || GV.getName().startswith("__sizedstack_") || GV.getName().startswith("__safe_stack"))
      return false;

  if (GV.hasAttribute(llvm::Attribute::DisableSanitizerInstrumentation))
      return false;

  // From (HW)ASan:
  if (GV.isDeclarationForLinker() || GV.getName().startswith("llvm."))
      return false;

  // Common symbols can't have aliases point to them, so they can't be tagged.
  if (GV.hasCommonLinkage())
      return false;

  // Globals with custom sections may be used in __start_/__stop_ enumeration,
  // which would be broken both by adding tags and potentially by the extra
  // padding/alignment that we insert.
  if (GV.hasSection())
      return false;

  // No point in instrumenting unused globals.
  if (GV.getNumUses() == 0)
      return false;

  // Only instrument globals that will not be defined in other modules.
  if (!GV.hasInitializer())
      return false;
  if (GV.getLinkage() != GlobalVariable::ExternalLinkage &&
      GV.getLinkage() != GlobalVariable::PrivateLinkage &&
      GV.getLinkage() != GlobalVariable::InternalLinkage)
      return false;

  // Two problems with thread-locals:
  // - The address of the main thread's copy can't be computed at link-time.
  // - Need to poison all copies, not just the main thread's one.
  if (GV.isThreadLocal()) return false;

  // Typeinfo pointers have to remain globals for codegen to work.
  // This also includes vtables (_ZTV*) which are assumed to be safe
  // because they are compiler-generated.
  // FCG: we need vtables to be instrumented to avoid slow checks.
  if (GV.getName().startswith("_ZT") && !GV.getName().startswith("_ZTV"))
      return false;

  if (GV.hasSection()) {
      StringRef Section = GV.getSection();

      // Globals from llvm.metadata aren't emitted, do not instrument them.
      if (Section == "llvm.metadata")
          return false;

      // Do not instrument globals from special LLVM sections.
      if (Section.find("__llvm") != StringRef::npos ||
          Section.find("__LLVM") != StringRef::npos) {
          return false;
      }

      // Do not instrument function pointers to initialization and termination
      // routines: dynamic linker will not properly handle redzones.
      if (Section.startswith(".preinit_array") ||
          Section.startswith(".init_array") ||
          Section.startswith(".fini_array") ||
          Section.startswith(".CRT")) {
          return false;
      }

      // TODO: generate a separate global namespace per section
      errs() << "ignoring global with custom section: " << GV << "\n";
      return false;
  }

  // TODO: skip safe globals (look at uses)
  // TODO: skip globals that are only used in noinstrument functions

  return true;
}

static bool stripDebugInfoRecursive(Function &F, SmallPtrSetImpl<Function*> &Visited) {
    if (Visited.count(&F))
        return false;
    Visited.insert(&F);
    bool Changed = stripDebugInfo(F);
    if (Changed) {
        for (Instruction &I : instructions(F)) {
            auto *CB = dyn_cast<CallBase>(&I);
            if (CB && CB->getCalledFunction())
                stripDebugInfoRecursive(*CB->getCalledFunction(), Visited);
        }
    }
    return Changed;
}

static bool stripDebugInfoRecursive(Function &F) {
    SmallPtrSet<Function*, 4> Visited;
    return stripDebugInfoRecursive(F, Visited);
}

Function *getOrInsertNoInstrumentFunction(Module &M, StringRef Name, FunctionType *Ty) {
    std::string FullName(NOINSTRUMENT_PREFIX);
    FullName += Name;
    if (Function *F = M.getFunction(FullName)) {
        if (F->getFunctionType() != Ty) {
            errs() << "unexpected type for helper function " << FullName << "\n";
            errs() << "  expected: " << *Ty << "\n";
            errs() << "  found:    " << *F->getFunctionType() << "\n";
            exit(1);
        }
        stripDebugInfoRecursive(*F);
        return F;
    }
    return Function::Create(Ty, GlobalValue::ExternalLinkage, FullName, &M);
}


namespace llvm {

STATISTIC(NumFunctions, "Total number of functions");
STATISTIC(NumUnsafeStackFunctions, "Number of functions with unsafe stack");
STATISTIC(NumUnsafeStackRestorePointsFunctions,
          "Number of functions that use setjmp or exceptions");

STATISTIC(NumAllocas, "Total number of allocas");
STATISTIC(NumUnsafeStaticAllocas, "Number of unsafe static allocas");
STATISTIC(NumUnsafeDynamicAllocas, "Number of unsafe dynamic allocas");
STATISTIC(NumUnsafeByValArguments, "Number of unsafe byval arguments");
STATISTIC(NumUnsafeStackRestorePoints, "Number of setjmps and landingpads");

STATISTIC(NumUnsafeStacks, "Total number of unsafe stacks");
STATISTIC(NumUnsafeStacksPerFunction, "Maximum number of unsafe stacks per function");
STATISTIC(NumDynAllocasMovedToHeap, "Number of dynamic allocas moved to the heap");
static const std::string kUnsafeStackPtrCountVar         = "__sizedstack_count";
static const std::string kUnsafeStackPtrVar              = "__sizedstack_ptrs";
static const std::string kUnsafeStackPtrVarFinal         = "__sizedstack_ptrs_final";
static const std::string kUnsafeStackPtrVarTemp          = "__sizedstack_ptrs_temp";
static const std::string kUnsafeStackSizeClassesVar      = "__sizedstack_sizeclasses";
static const std::string kUnsafeStackSizeClassesVarFinal = "__sizedstack_sizeclasses_final";

STATISTIC(NInstrumented, "Number of instrumented globals");
STATISTIC(NSkipped,      "Number of skipped safe globals");
STATISTIC(NOrigBytes,    "Previous allocation size for instrumented globals");
STATISTIC(NTotalBytes,   "Total allocation size for globals (globals + padding + redzones)");
STATISTIC(NRedzoneBytes, "Redzone bytes in global allocation");
STATISTIC(NPaddingBytes, "Wasted padding bytes in global allocation due to alignment");


static const unsigned kDefaultUnsafeStackSize = 0x2800000;

// Multiples of 2 size classes
static const std::vector<uint64_t> kSizeClasses = {
  16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288
};


// fuzzing two stage LTO management
cl::opt<bool> ClDelayInstrumentation("san-delay-instrumentation",
                                       cl::desc("make sure function marking is applied but do not instrument yet"),
                                       cl::Hidden, cl::init(false));

cl::opt<int> OptRedzoneValue("redzone-value",
    cl::desc("Guard value for redzone initalization"),
    cl::init(0));

// static const unsigned kMaxAlignment = 128;

cl::opt<unsigned long long> OptRedzoneSize("redzone-size",
    cl::desc("Redzone size"),
    cl::init(32));

cl::opt<bool> OptNamedTypes("global-redzones-named-types",
        cl::desc("Create named instead of literal types for global replacement structs"),
        cl::init(false));

#ifndef USE_GOLD_PASSES
static cl::opt<bool> ClSizedStack(
    "sized-stack",
    cl::desc("If set will be enabled"),
    cl::init(true));
#endif
static cl::opt<bool> OptInitRedzones("stack-init-redzones",
    cl::desc("Initialize stack redzones upon allocation"),
    cl::init(false));
static cl::opt<bool> OptStackShadowMem("stack-shadowmem",
    cl::desc("Maintain shadow memory on stack allocation"),
    cl::init(false));
static cl::opt<bool> OptOnlyNoDynAllocas("only-no-dyn-allocas",
    cl::desc("Only move dynamic allocas to the heap, don't do anything with static allocas"),
    cl::init(false));

static cl::opt<bool> ClSubsanCcmpFastPath(
    "subsan-ccmp-fast-path", cl::init(true), cl::Hidden,
    cl::desc("Subsan: pre-compute tag in fast path so LLVM emits ccmp+bne. "
             "false: compute tag only in slow path (cbz tag, fall-through)."));

static cl::opt<unsigned> ClSubsanInlineTagUnrollLimit(
    "subsan-inline-tag-unroll-limit", cl::init(8), cl::Hidden,
    cl::desc("Subsan: inline tag_memory up to N granules (16B each); larger "
             "allocas fall back to runtime call."));
static cl::opt<bool> ClSubsanInlineTagMemory(
    "subsan-inline-tag-memory", cl::init(true), cl::Hidden,
    cl::desc("Subsan: emit stack alloca shadow tag/untag as inline IR instead "
             "of __subsan_tag_memory function call."));

static cl::opt<bool> ClSubsanFixedShadowBase(
    "subsan-fixed-shadow-base", cl::init(false), cl::Hidden,
    cl::desc("Subsan: use compile-time fixed shadow base address (immediate "
             "constant) instead of runtime GV load. Requires runtime to mmap "
             "the shadow region at exactly subsan-shadow-offset (MAP_FIXED)."));

static cl::opt<uint64_t> ClSubsanShadowOffset(
    "subsan-shadow-offset", cl::init(0x200000000ULL), cl::Hidden,
    cl::desc("Subsan: fixed shadow base address. Must match the runtime "
             "InitShadow mmap base. Default 0x200000000 (8 GB) matches the "
             "first hint of subsan_runtime.cpp InitShadowImpl()."));

static cl::opt<bool> ClSubsanPrefetchShadow(
    "subsan-prefetch-shadow", cl::init(false), cl::Hidden,
    cl::desc("Subsan: emit `prfm pldl1keep, [next_granule_shadow]` after each "
             "shadow load. Sequential access pattern 의 latency 감소 시도."));

static cl::opt<bool> ClSubsanLamBmi2Bextr(
    "subsan-lam-bmi2-bextr", cl::init(false), cl::Hidden,
    cl::desc("Subsan x86 LAM: BMI2 bextr 로 granule_idx 추출을 1 inst 화 "
             "(ARM ubfx 등가). -mbmi2 필요."));

static cl::opt<bool> ClSubsanInlineCheckBarrier(
    "subsan-inline-check-barrier", cl::init(false), cl::Hidden,
    cl::desc("Subsan: insert inline asm barrier on Meta load result to "
             "prevent LTO Reassociate from folding inline check sub with "
             "algorithm chain. Experimental — chain merge ROI 불확실."));

static cl::opt<bool> ClSubsanMetadataSharing(
    "subsan-metadata-sharing", cl::init(false), cl::Hidden,
    cl::desc("Subsan: enable function-level metadata sharing (cross-BB same-base "
             "shadow load deduplication). May add cmp/branch overhead. "
             "Beneficial for cache-bound workloads (ref) where shadow load "
             "is hot."));

static cl::opt<bool> ClSubsanMoveArgvToHeap(
    "subsan-move-argv-to-heap", cl::init(true), cl::Hidden,
    cl::desc("Subsan: at main entry, call __subsan_move_argv_to_heap(argc, "
             "argv) and replace argv uses with the tagged result."
             "Avoids slow-path cbz resume on untagged argv accesses."));

static cl::opt<bool> ClSubsanHoistShadowBase(
    "subsan-hoist-shadow-base", cl::init(true), cl::Hidden,
    cl::desc("Subsan: hoist __hwasan_shadow_memory_dynamic_address load to "
             "function entry. Each InsertCheck reuses cached SSA value "
             "instead of emitting per-BB ldr. HWASan-style."));

static cl::opt<bool> ClSubsanMallocAlign(
    "subsan-malloc-align", cl::init(true), cl::Hidden,
    cl::desc("Subsan: attach align(16) + noalias attributes to "
             "__subsan_malloc/calloc/realloc wrapper. Goal: improve GVN/CSE "
             "of shadow loads for adjacent byte accesses."));

static cl::opt<bool> ClSubsanUbBarrier(
    "subsan-ub-barrier", cl::init(false), cl::Hidden,
    cl::desc("Subsan: emit inline asm barrier on Ub to prevent LLVM "
             "Reassociate from splitting `Meta - (ptr + size-1)` into "
             "`(Meta - ptr) - (size-1)`. Goal: chain depth -1 cycle."));

} // namespace llvm

static void emitInlineTagMemory(IRBuilder<> &IRB, Module *M,
                                  llvm::Value *AddrInt, uint64_t ExactSize,
                                  uint8_t ConstTag) {
  llvm::LLVMContext &C = M->getContext();
  llvm::Type *I64Ty = llvm::Type::getInt64Ty(C);
  llvm::Type *I64PtrTy = llvm::PointerType::get(I64Ty, 0);
  llvm::Type *I8Ty = llvm::Type::getInt8Ty(C);
  if (ExactSize == 0) return;
  uint64_t AlignedSize = (ExactSize + 15) & ~15ULL;
  uint64_t GranuleCount = AlignedSize / 16;

  if (GranuleCount > ClSubsanInlineTagUnrollLimit) {
    llvm::FunctionCallee TagMemoryFn = M->getOrInsertFunction(
        "__subsan_tag_memory", I64Ty, I64Ty, I64Ty, I8Ty);
    IRB.CreateCall(TagMemoryFn,
        {AddrInt, llvm::ConstantInt::get(I64Ty, ExactSize),
         llvm::ConstantInt::get(I8Ty, ConstTag)});
    return;
  }

  llvm::Value *ShadowBase;
  if (ClSubsanFixedShadowBase) {
    ShadowBase = llvm::ConstantInt::get(I64Ty, ClSubsanShadowOffset);
  } else {
    llvm::Constant *ShadowBaseC = M->getOrInsertGlobal(
        "__hwasan_shadow_memory_dynamic_address", I64Ty);
    llvm::GlobalVariable *ShadowBaseGV =
        llvm::dyn_cast<llvm::GlobalVariable>(ShadowBaseC);
    if (ShadowBaseGV && !ShadowBaseGV->hasInitializer())
      ShadowBaseGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    ShadowBase = IRB.CreateAlignedLoad(
        I64Ty, ShadowBaseGV, llvm::Align(8), "subsan.tagmem.shadow_base");
    llvm::cast<llvm::LoadInst>(ShadowBase)->setMetadata(
        llvm::LLVMContext::MD_invariant_load,
        llvm::MDNode::get(C, std::nullopt));
  }

  // granule_start = (addr >> 4) & 0x000FFFFFFFFFFFFF
  llvm::Value *Shifted = IRB.CreateLShr(
      AddrInt, llvm::ConstantInt::get(I64Ty, 4));
  llvm::Value *GranuleStart = IRB.CreateAnd(
      Shifted, llvm::ConstantInt::get(I64Ty, subsanGranuleIdxMask),
      "subsan.tagmem.granule_start");
  llvm::Value *ShadowI64Base = IRB.CreateIntToPtr(ShadowBase, I64PtrTy);
  llvm::Value *ShadowPtr = IRB.CreateInBoundsGEP(
      I64Ty, ShadowI64Base, GranuleStart, "subsan.tagmem.shadow_ptr");

  llvm::Value *Meta;
  if (ConstTag == 0) {
    Meta = llvm::ConstantInt::get(I64Ty, 0);
  } else {
    llvm::Value *End = IRB.CreateAdd(
        AddrInt, llvm::ConstantInt::get(I64Ty, ExactSize - 1),
        "subsan.tagmem.end");
    Meta = IRB.CreateOr(End,
        llvm::ConstantInt::get(I64Ty, uint64_t(ConstTag) << subsanTagShift),
        "subsan.tagmem.meta");
  }

  unsigned SwiftsanKindId = M->getMDKindID("swiftsan");
  llvm::MDNode *SwiftsanMD = llvm::MDNode::get(C, std::nullopt);
  for (uint64_t i = 0; i < GranuleCount; i++) {
    llvm::Value *SlotPtr = (i == 0)
        ? ShadowPtr
        : IRB.CreateInBoundsGEP(I64Ty, ShadowPtr,
                                llvm::ConstantInt::get(I64Ty, i));
    auto *Store = IRB.CreateAlignedStore(Meta, SlotPtr, llvm::Align(8));
    Store->setMetadata(SwiftsanKindId, SwiftsanMD);
  }
}

/// Use __safestack_pointer_address even if the platform has a faster way of
/// access safe stack pointer.
static cl::opt<bool>
    SafeStackUsePointerAddress("safestack-use-pointer-address",
                                  cl::init(false), cl::Hidden);

static cl::opt<bool> ClColoring("safe-stack-coloring",
                                cl::desc("enable safe stack coloring"),
                                cl::Hidden, cl::init(false));

namespace {

/// Rewrite an SCEV expression for a memory access address to an expression that
/// represents offset from the given alloca.
///
/// The implementation simply replaces all mentions of the alloca with zero.
class AllocaOffsetRewriter : public SCEVRewriteVisitor<AllocaOffsetRewriter> {
  const Value *AllocaPtr;

public:
  AllocaOffsetRewriter(ScalarEvolution &SE, const Value *AllocaPtr)
      : SCEVRewriteVisitor(SE), AllocaPtr(AllocaPtr) {}

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    if (Expr->getValue() == AllocaPtr)
      return SE.getZero(Expr->getType());
    return Expr;
  }
};

static inline uint64_t round_to_power_of_two(uint64_t n) {
    // assumes n is not bigger than 32-bit size
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

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

  void print(raw_ostream &O) {
    O << "Mem" << (IsWrite ? "Write(" : "Read(");
    O << "inst={ " << *getInsn() << " }";
    O << " size={ " << TypeSize << " }";
    O << " align=" << Alignment.valueOrOne().value();
    O << ")";
  }
  const std::string toString() {
    std::string s;
    raw_string_ostream ss(s);
    print(ss);
    return s;
  }
};

class InvPathConditions {
  public:
  std::set<Value *> TrueConditions;
  std::set<Value *> FalseConditions;

  // pair = (switch-val, match-val)  
  // represents: if switch-val == match-val
  std::set<std::pair<Value *, Value *>> SwitchConditions;

  InvPathConditions(std::set<Value *> trues, std::set<Value *> falses, std::set<std::pair<Value *, Value *>> switches)
  : TrueConditions(trues), FalseConditions(falses), SwitchConditions(switches){

  }

  InvPathConditions(){

  }

  unsigned get_cond_count(){
    return TrueConditions.size() + FalseConditions.size() + SwitchConditions.size();
  }
};

class LoopRangeCheck {
  public:
  Instruction* InsertPt;
  Type* ptrType;
  const SCEV *ScStart;
  const SCEV *ScEnd;
  Instruction *MemAccess;
  Loop *L;
  // bool conditional;
  // InvPathConditions BestConds;
  bool invariant;
  Value *Ptr;
  bool reuseMeta;

  LoopRangeCheck(Instruction *I, Type *Ty, const SCEV *start, const SCEV *end, Instruction *src, Loop *loop, /*bool b, InvPathConditions conds,*/ bool inv, Value *invP, bool remeta)
  : InsertPt(I), ptrType(Ty), ScStart(start), ScEnd(end), MemAccess(src), L(loop), /*conditional(b), BestConds(conds),*/ invariant(inv), Ptr(invP), reuseMeta(remeta){

  }
};

class OffsetSameBase {
  public:
  InterestingMemoryOperand Oper;
  Instruction *Addr;
  APInt Offset;
  bool smaller;

  OffsetSameBase(InterestingMemoryOperand &O, Instruction* IAddr, APInt Off, bool s)
  : Oper(O), Addr(IAddr), Offset(Off), smaller(s){
  }

};

class CheckMergeCands {
  public:
  InterestingMemoryOperand TopTarget;
  SmallVector<InterestingMemoryOperand, 16> ContainedOps;
  OffsetSameBase Smallest;
  OffsetSameBase Biggest;

  CheckMergeCands(InterestingMemoryOperand &Oper, SmallVector<InterestingMemoryOperand, 16> CO, OffsetSameBase S, OffsetSameBase B)
  : TopTarget(Oper), ContainedOps(CO), Smallest(S), Biggest(B){

  }
};

class NegPosPair {
  public:
  InterestingMemoryOperand Neg;
  InterestingMemoryOperand Pos;
  bool neg_first;

  NegPosPair(InterestingMemoryOperand &N, InterestingMemoryOperand &P, bool b) : Neg(N), Pos(P), neg_first(b){

  }
};

class MinMaxPair {
  public:
  InterestingMemoryOperand First;
  InterestingMemoryOperand Second;

  MinMaxPair(InterestingMemoryOperand &F, InterestingMemoryOperand &S) : First(F), Second(S) {

  }
};

class MinMaxChain {
  public:
  InterestingMemoryOperand First;
  InterestingMemoryOperand Last;
  SmallVector<InterestingMemoryOperand, 16> ContainedOps;
  unsigned sz;
  PostDominatorTree &PDT;
  
  MinMaxChain(InterestingMemoryOperand &F, InterestingMemoryOperand &L, PostDominatorTree &PDT) : First(F), Last(L), PDT(PDT) {
    ContainedOps.clear();
    sz = 1;
  }

  MinMaxChain(InterestingMemoryOperand &F, InterestingMemoryOperand &L, PostDominatorTree &PDT, SmallVector<InterestingMemoryOperand, 16> cont) : First(F), Last(L), PDT(PDT) {
    ContainedOps.clear();
    sz = 1;
    for(auto entry : cont){
      ContainedOps.push_back(entry);
      sz++;
    }
  }

  void add(InterestingMemoryOperand &O){
    if(this->PDT.dominates(O.getInsn(), this->Last.getInsn())){
      this->ContainedOps.push_back(this->Last);
      this->Last = O;
    }
    else{
      this->ContainedOps.push_back(O);
    }
    this->sz++;
  }

};

class GlobalRedzones {
    static char ID;

private:
    SmallVector<uint64_t, 64> SizeClasses;
    const DataLayout *DL;
    IntegerType *Int8Ty;
    Type *Int64Ty; 

public:
    GlobalRedzones() : SizeClasses(kSizeClasses.begin(), kSizeClasses.end()) {}

    bool runOnModule(Module &M) {
        Int8Ty = IntegerType::get(M.getContext(), 8);
        Int64Ty = Type::getInt64Ty(M.getContext());
        DL = &M.getDataLayout();

        // Create an array of globals for each sizeclass, appending a redzone
        // to each global with optional padding. Group the arrays in a wrapping
        // array that adds the rightmost redzone. Separate constants from
        // mutable variables in a separate outer array. Alignment is enforced
        // by (1) aligning the outer array to the maximum alignment of all
        // globals in the array and (2) aligning each sizeclass array to the
        // maximum alignment of its members. The latter is done by predecing it
        // with padding bytes:
        // @glob.const.X = constant { redzone, { alignment, { glob, [padding,] redzone }, .. }, .. }
        // @glob.mut.X = { ... }
        MapVector<uint64_t, SmallVector<GlobalVariable*, 8>> WorkListC, WorkListM;

        for (GlobalVariable &GV : M.globals()) {
            if (shouldInstrumentGlobal(GV)){
              (GV.isConstant() ? WorkListC : WorkListM)[getSizeClass(GV)].push_back(&GV);
            }
            else{
                ++NSkipped;
            }
        }

        auto SortedC = WorkListC.takeVector();
        std::sort(SortedC.begin(), SortedC.end());
        auto SortedM = WorkListM.takeVector();
        std::sort(SortedM.begin(), SortedM.end());
        subsanInstrumentGlobalsHWAsanStyle<8, 8>(M, SortedC, SortedM);

        // Compute wasted space
        // double Ratio = (NTotalBytes - NOrigBytes) * 100 / (double)NOrigBytes;
        // errs() << "increased globals size from " << NOrigBytes << " to " << NTotalBytes <<
                //  " bytes (by " << format("%.0f", Ratio) << "%)" << "\n";

        return !SortedC.empty() || !SortedM.empty();
    }

private:
    uint64_t getGlobalRedzoneSize(uint64_t Size) {
      // ASan global redzone sizes
      constexpr uint64_t kMaxRZ = 1 << 18;
      const uint64_t MinRZ = 32;
      uint64_t RZ = 0;
      if (Size <= MinRZ / 2) {
        // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
        // at least 32 bytes, optimize when Size is less than or equal to
        // half of MinRZ.
        RZ = MinRZ - Size;
      } else {
        // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
        RZ = std::clamp((Size / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);
      }
      return RZ;
    }

    uint64_t getSizeClass(GlobalVariable &GV) {
        uint64_t Size = DL->getTypeAllocSize(GV.getValueType());
        unsigned Align = GV.getAlignment();

        // Allocate redzone
        uint64_t RZ = getGlobalRedzoneSize(Size);
        Size += RZ;

        // Find the first size class that fits and is a multiple of the
        // global's alignment
        for (uint64_t SC : SizeClasses) {
            if (SC > RZ)
                if (Size <= SC && (!Align || SC % Align == 0))
                    return SC;
        }

        // For very big objects, round up size to alignment and create a new
        // power of two size class
        Size = round_to_power_of_two(Size+Align);
        errs() << "add new global size class " << Size << "\n";
        SizeClasses.push_back(Size);
        return Size;
    }

    Constant *repeatByte(int ByteVal, unsigned N) {
        ArrayType *Ty = ArrayType::get(Int8Ty, N);
        SmallVector<Constant*, 16> Bytes;
        Constant *Byte = ConstantInt::get(Int8Ty, ByteVal);
        for (unsigned i = 0; i < N; ++i)
            Bytes.push_back(Byte);
        return ConstantArray::get(Ty, Bytes);
    }

    GlobalVariable *getNoInstrumentGlobal(Module &M, StringRef Name, bool AllowMissing) {
        std::string FullName(NOINSTRUMENT_PREFIX);
        FullName += Name;
        GlobalVariable *GV = M.getNamedGlobal(FullName);
        if (!GV && !AllowMissing) {
            errs() << "Error: could not find helper global " << FullName << "\n";
            exit(1);
        }
        return GV;
    }

    template<unsigned NConst, unsigned NMut>
    void subsanInstrumentGlobalsHWAsanStyle(
        Module &M,
        const std::vector<std::pair<uint64_t, SmallVector<GlobalVariable*, NConst>>> &ConstWL,
        const std::vector<std::pair<uint64_t, SmallVector<GlobalVariable*, NMut>>> &MutWL) {
      LLVMContext &C = M.getContext();
      const DataLayout &DL = M.getDataLayout();
      Type *I64Ty = Type::getInt64Ty(C);
      Type *I8Ty = Type::getInt8Ty(C);
      Type *VoidTy = Type::getVoidTy(C);

      // runtime API: void __subsan_init_global(uintptr_t addr, uintptr_t size, uint8_t tag)
      FunctionCallee InitGlobalFn = M.getOrInsertFunction(
          "__subsan_init_global", VoidTy, I64Ty, I64Ty, I8Ty);

      SmallVector<GlobalVariable *, 32> Globals;
      for (auto &SCPair : ConstWL)
        for (GlobalVariable *GV : SCPair.second)
          Globals.push_back(GV);
      for (auto &SCPair : MutWL)
        for (GlobalVariable *GV : SCPair.second)
          Globals.push_back(GV);
      if (Globals.empty()) return;

      FunctionType *CtorFnTy = FunctionType::get(VoidTy, {}, false);
      Function *CtorFn = Function::Create(
          CtorFnTy, GlobalValue::InternalLinkage,
          "__subsan_module_globals_init", &M);
      BasicBlock *CtorBB = BasicBlock::Create(C, "entry", CtorFn);
      IRBuilder<> IRB(CtorBB);

      // [subsan x86_64 LAM port] tag namespace per-arch — set in runOnModule().
      //   AArch64 TBI    : globals 86..170
      //   x86_64 LAM_U57 : globals 22..42
      uint8_t Tag = subsanGlobalTagLo;
      for (GlobalVariable *GV : Globals) {
        uint64_t SizeInBytes = DL.getTypeAllocSize(GV->getValueType());
        if (SizeInBytes == 0) continue;
        if (!GV->hasInitializer()) continue;

        Constant *Initializer = GV->getInitializer();
        uint64_t NewSize = (SizeInBytes + 15) & ~15ULL;
        if (SizeInBytes != NewSize) {
          std::vector<uint8_t> PadBytes(NewSize - SizeInBytes, 0);
          Constant *Padding = ConstantDataArray::get(C, PadBytes);
          Initializer = ConstantStruct::getAnon({Initializer, Padding});
        }

        auto *NewGV = new GlobalVariable(
            M, Initializer->getType(), GV->isConstant(),
            GlobalValue::ExternalLinkage, Initializer,
            GV->getName() + ".subsan");
        NewGV->copyAttributesFrom(GV);
        NewGV->setLinkage(GlobalValue::PrivateLinkage);
        NewGV->copyMetadata(GV, 0);
        NewGV->setAlignment(
            std::max(GV->getAlign().valueOrOne(), Align(16)));

        NewGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

        Value *NewGVAsInt = IRB.CreatePtrToInt(NewGV, I64Ty);
        IRB.CreateCall(InitGlobalFn,
                       {NewGVAsInt, ConstantInt::get(I64Ty, SizeInBytes),
                        ConstantInt::get(I8Ty, Tag)});

        Constant *Aliasee = ConstantExpr::getIntToPtr(
            ConstantExpr::getAdd(
                ConstantExpr::getPtrToInt(NewGV, I64Ty),
                ConstantInt::get(I64Ty, uint64_t(Tag) << subsanTagShift)),
            GV->getType());
        auto *Alias = GlobalAlias::create(
            GV->getValueType(), GV->getAddressSpace(),
            GV->getLinkage(), "", Aliasee, &M);
        Alias->setVisibility(GV->getVisibility());
        Alias->takeName(GV);

        GV->replaceAllUsesWith(Alias);
        GV->eraseFromParent();

        // global namespace cycle (range per-arch, set in runOnModule).
        Tag = (Tag >= subsanGlobalTagHi) ? subsanGlobalTagLo : (Tag + 1);
      }
      IRB.CreateRetVoid();

      appendToGlobalCtors(M, CtorFn, /*Priority=*/1);
    }

    template<unsigned N>
    void instrumentGlobals(Module &M, bool isConstant,
            const std::vector<std::pair<uint64_t, SmallVector<GlobalVariable*, N>>> &GlobalsBySizeClass) {
        const std::string Name = std::string("glob.") + (isConstant ? "const" : "mut");

        for (auto &it : GlobalsBySizeClass) {
            const uint64_t SizeClass = it.first;
            const SmallVectorImpl<GlobalVariable*> &GVs = it.second;

            // add one SizeClass sized object (keeps alignment intact)
            SmallVector<Constant*, 16> SCInits;
            Constant* FakeObjectUnderflowSC = repeatByte(0, SizeClass);
            SCInits.push_back(FakeObjectUnderflowSC);
            unsigned EasiestSplit = SizeClass / 2;
            Constant *FakeObjFiller = repeatByte(0, EasiestSplit);
            Constant *FakeRedzone = repeatByte(0, EasiestSplit - 8);
            // use the first FinalGlobalEndOfObjs entry
            Constant *SecondFakeObjInit = getStruct({FakeObjFiller, FakeRedzone, ConstantInt::get(Int64Ty, 0x42)}, Name, /*isNameOptional=*/true);
            SCInits.push_back(SecondFakeObjInit);

            for (GlobalVariable *GV : GVs){
                // placeholder metadata size until we know the true addr location
                Constant *ObjWithRedzone = padAndAppendRedzone(GV, SizeClass, ConstantInt::get(Int64Ty, 0x42));
                SCInits.push_back(ObjWithRedzone);
            }
            
            std::string SCName = Name + "." + std::to_string(SizeClass) + ".ty";
            SmallVector<Type*, 8> ElTys;
            for (Constant *El : SCInits)
              ElTys.push_back(El->getType());
            StructType *SCTy = StructType::create(ElTys, SCName, /*isPacked=*/true);

            // Create the global
            GlobalVariable *ReplClass = new GlobalVariable(M, SCTy, isConstant,
                    GlobalValue::InternalLinkage, ConstantStruct::get(SCTy, SCInits), Name);
            ReplClass->setAlignment(Align(SizeClass));

            if(isImplicitTagging){
              // for implicit tagging, we move the size class into a specific section
              // which a linker script will then place at the right address
              std::string sec = std::string("basebounds_section_") + (isConstant ? "const_" : "mut_");
              sec += std::to_string(SizeClass);

              ReplClass->setSection(sec);
            }

            // Create index array for replacement GEPs below
            Type *Int32Ty = IntegerType::get(M.getContext(), 32);
            Constant *Idx[] = {
                ConstantInt::get(Int32Ty, 0), // get the value of the big new global
                nullptr,                      // index of wrapped global in sizeclass array
                ConstantInt::get(Int32Ty, 0)  // the global is first in [global, padding, redzone]
            };

            SmallVector<Constant*, 16> FinalGlobalEndOfObjs;
            SmallVector<Constant*, 16> GVToReplace;

            uint64_t tag = (uint64_t)(log2(SizeClass)) << subsanTagShift;
            Type *GEPTy = ReplClass->getValueType();
            // Type *Int64Ty = Type::getInt64Ty(M.getContext());
            int GlobIndex = 2;
            for (GlobalVariable *GV : GVs) {
                Idx[1] = ConstantInt::get(Int32Ty, GlobIndex++);
                Constant *newGlobPtr = ConstantExpr::getInBoundsGetElementPtr(GEPTy, ReplClass, Idx);

                // by default the aliasee is the new global (GEP entry in the class)
                // but if we perform explicit pointer tagging, the aliasee should
                // include the tag offset on the pointer
                Constant *Aliasee = newGlobPtr;
                if(!isImplicitTagging){
                  // Source for replacing with tagged globals: HWASan
                  Aliasee = ConstantExpr::getIntToPtr(
                      ConstantExpr::getAdd(
                          ConstantExpr::getPtrToInt(newGlobPtr, Int64Ty),
                          ConstantInt::get(Int64Ty, uint64_t(tag))),
                      GV->getType());
                }

                uint64_t GlobalTrueObjSize = DL->getTypeAllocSize(GV->getInitializer()->getType());

                Constant *EndOfObj = ConstantExpr::getAdd(
                  ConstantExpr::getPtrToInt(Aliasee, Int64Ty),
                  ConstantInt::get(Int64Ty, GlobalTrueObjSize),
                  Int64Ty
                );

                FinalGlobalEndOfObjs.push_back(EndOfObj);

                // Create an Alias such that we can do this constant expression once in
                // the global definition instead of having to put the constexpr at every use site.
                auto *Alias = GlobalAlias::create(
                    GV->getType(), GV->getAddressSpace(), GV->getLinkage(),
                    "", Aliasee, &M);
                Alias->setVisibility(GV->getVisibility());
                Alias->takeName(GV);

                GVToReplace.push_back(Alias);
            }

            // now construct the initializers by linking the end of obj to the prev obj redzone
            SmallVector<Constant*, 16> SCInitsWithMeta;
            // first fake obj (for underflow detection)
            SCInitsWithMeta.push_back(FakeObjectUnderflowSC);
            // second fake obj: metadata for first real object
            // use the first FinalGlobalEndOfObjs entry
            SecondFakeObjInit = getStruct({FakeObjFiller, FakeRedzone, FinalGlobalEndOfObjs[0]}, Name, /*isNameOptional=*/true);
            SCInitsWithMeta.push_back(SecondFakeObjInit);

            unsigned NextGlobal = 1;
            Constant* SizeMeta = NULL;
            for (GlobalVariable *GV : GVs) {
                if(NextGlobal < FinalGlobalEndOfObjs.size()){
                  SizeMeta = FinalGlobalEndOfObjs[NextGlobal];
                }
                else{
                  // the last object can have any metadata value, just set zero for best detection
                  SizeMeta = ConstantInt::get(Int64Ty, 0x0);
                }
                Constant *ObjWithRedzone = padAndAppendRedzone(GV, SizeClass, SizeMeta);
                SCInitsWithMeta.push_back(ObjWithRedzone);
                NextGlobal++;
            }

            // final update to the initializer with end of obj addrs in metadata
            Constant *newInitMeta = ConstantStruct::get(SCTy, SCInitsWithMeta);
            ReplClass->setInitializer(newInitMeta);

            // finally: replace the GVs
            unsigned idx = 0;
            for (GlobalVariable *GV : GVs) {
                GV->replaceAllUsesWith(GVToReplace[idx]);
                GV->eraseFromParent();
                idx++;
            }
            SCInits.clear();
            SCInitsWithMeta.clear();
        }
    }

    Constant *padAndAppendRedzone(GlobalVariable *GV, unsigned SizeClass, Constant *SizeMeta) {
        uint64_t Size = DL->getTypeAllocSize(GV->getValueType());
        uint64_t RZ = getGlobalRedzoneSize(Size);
        Constant *C = GV->getInitializer();
        Type *Ty = C->getType();
        assert(DL->getTypeAllocSize(C->getType()) + RZ <= SizeClass);
        assert(DL->getTypeAllocSize(C->getType()) == Size);
        std::string Name = GV->getName().str() + ".wrap";
        unsigned PadBytes = SizeClass - RZ - DL->getTypeAllocSize(Ty);
        // Constant *SizeMeta = ConstantInt::get(Int64Ty, DL->getTypeAllocSize(Ty));
        Constant *Redzone = repeatByte(0, RZ - 8);

        if (PadBytes) {
            Constant *Padding = repeatByte(0, PadBytes);
            C = getStruct({C, Padding, Redzone, SizeMeta}, Name + ".pad", /*isNameOptional=*/true);
            NPaddingBytes += PadBytes;
        } else {
            C = getStruct({C, Redzone, SizeMeta}, Name, /*isNameOptional=*/true);
        }
        NRedzoneBytes += RZ;
        assert(DL->getTypeAllocSize(C->getType()) == SizeClass);
        return C;
    }

    static Constant *getStruct(ArrayRef<Constant*> Els, StringRef TyName,
            bool isNameOptional = false, bool isPacked = true) {
        if (isNameOptional && !OptNamedTypes)
            return ConstantStruct::getAnon(Els, isPacked);

        SmallVector<Type*, 8> ElTys;
        for (Constant *El : Els)
            ElTys.push_back(El->getType());
        StructType *Ty = StructType::create(ElTys, TyName, isPacked);
        return ConstantStruct::get(Ty, Els);
    }

    static Constant *getArray(ArrayRef<Constant*> Els, Type *ElTy) {
        return ConstantArray::get(ArrayType::get(ElTy, Els.size()), Els);
    }

};

class SizedStackRuntime {
  GlobalVariable *stackPointerArray;

  StringMap<size_t> typeIndexByTypeId;
  size_t typeIndexNext;
  SmallVector<uint64_t, 16> AssignedSizeClasses;

  Module &M;
  const DataLayout &DL;

  PointerType *StackPtrTy;
  IntegerType *IntPtrTy;
  IntegerType *Int64Ty;

  static std::string getStaticStackID(uint64_t Size, const Value &V);
  static uint64_t IncludeRedzoneSize(uint64_t Size);
  static uint64_t roundUpToSizeClass(uint64_t Size);


  GlobalVariable *createStackPtrArray(StringRef varName, size_t count);
  GlobalVariable *createStackPtrCount(StringRef varName, size_t count);
  GlobalVariable *createSizeClassArray(StringRef varName);

public:
  size_t getTypeIndex(StringRef StackID);

  /// Unsafe stack alignment. Each stack frame must ensure that the stack is
  /// aligned to this value. We need to re-align the unsafe stack if the
  /// alignment of any object on the stack exceeds this value.
  ///
  /// 16 seems like a reasonable upper bound on the alignment of objects that we
  /// might expect to appear on the stack on most common targets.
  enum { StackAlignment = 16 };

  Function *DynAllocFunc, *DynFreeFunc, *DynFreeOptFunc, *ShadowMemPoisonFunc;

  SizedStackRuntime(Module &M)
      : M(M), DL(M.getDataLayout()),
        StackPtrTy(Type::getInt8PtrTy(M.getContext())),
        IntPtrTy(DL.getIntPtrType(M.getContext())),
        Int64Ty(Type::getInt64Ty(M.getContext())) {}

  /// \brief Calculate the allocation size of a given alloca. Returns 0 if the
  /// size can not be statically determined.
  uint64_t getStaticAllocaAllocationSize(const AllocaInst* AI);

  bool initialize();
  bool finalize();

  std::string getStackID(const AllocaInst &AI);
  std::string getStackID(const Argument &Arg);

  Value *getOrCreateUnsafeStackPtr(IRBuilder<> &IRB, Function &F, StringRef typeId);

}; // class SizedStackRuntime


/// The SafeStack pass splits the stack of each function into the safe
/// stack, which is only accessed through memory safe dereferences (as
/// determined statically), and the unsafe stack, which contains all
/// local variables that are accessed in ways that we can't prove to
/// be safe.
class SafeStack {
  Function &F;
  const TargetLoweringBase &TL;
  const DataLayout &DL;
  DomTreeUpdater *DTU;
  ScalarEvolution &SE;
  DominatorTree &DT;
  DominanceFrontier &DF;
  SizedStackRuntime &RT;
  TargetLibraryInfo &TLI;
  LoopInfo &LI;
  AliasAnalysis *AA;
  // DependenceInfo *DI;

  Type *StackPtrTy;
  Type *IntPtrTy;
  Type *Int32Ty;
  Type *Int8Ty;

  Value *UnsafeStackPtr = nullptr;

  Value *FnShadowBase = nullptr;

  // for checks
  SmallVector<Instruction*, 16> SafeMemOps;
  FunctionCallee swiftsan_report_fn = nullptr;
  FunctionCallee swiftsan_argv_fn = nullptr;

  // Mem family functions (to avoid calling LLVM intrinsics)
  FunctionCallee SwiftsanMemmove, SwiftsanMemcpy, SwiftsanMemset;
  // Mem Transfers (check source only) -- for safe allocas
  FunctionCallee SwiftsanMemmoveSrc, SwiftsanMemcpySrc;
  // Mem Transfers (check dest only) -- for safe allocas
  FunctionCallee SwiftsanMemmoveDst, SwiftsanMemcpyDst;

  /// Unsafe stack alignment. Each stack frame must ensure that the stack is
  /// aligned to this value. We need to re-align the unsafe stack if the
  /// alignment of any object on the stack exceeds this value.
  ///
  /// 16 seems like a reasonable upper bound on the alignment of objects that we
  /// might expect to appear on the stack on most common targets.
  static constexpr Align StackAlignment = Align::Constant<16>();

  void MarkAllVarArgLoads(Function &F);
  void MarkErrnoAccesses(Function &F);

  /// Find all static allocas, dynamic allocas, return instructions and
  /// stack restore points (exception unwind blocks and setjmp calls) in the
  /// given function and append them to the respective vectors.
  void findInsts(Function &F, SmallVectorImpl<AllocaInst *> &StaticAllocas,
                 SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                 SmallVectorImpl<Argument *> &ByValArguments,
                 SmallVectorImpl<Instruction *> &Returns,
                 SmallVectorImpl<Instruction *> &StackRestorePoints);

  /// Calculate the allocation size of a given alloca. Returns 0 if the
  /// size can not be statically determined.
  uint64_t getStaticAllocaAllocationSize(const AllocaInst* AI);

  /// Allocate space for all static allocas in \p StaticAllocas,
  /// replace allocas with pointers into the unsafe stack.
  ///
  /// \returns A pointer to the top of the unsafe stack after all unsafe static
  /// allocas are allocated.
  Value *moveStaticAllocasToUnsafeStack(IRBuilder<> &IRB, Function &F,
                                        ArrayRef<AllocaInst *> StaticAllocas,
                                        ArrayRef<Argument *> ByValArguments,
                                        Instruction *BasePointer,
                                        AllocaInst *StackGuardSlot,
                                        Value *UnsafeStackPtr,
                                        StringRef StackID,
                                        SmallVectorImpl<std::tuple<Value*, uint64_t, uint8_t>> &TaggedStackAllocas);

  /// Generate code to restore the stack after all stack restore points
  /// in \p StackRestorePoints.
  ///
  /// \returns A local variable in which to maintain the dynamic top of the
  /// unsafe stack if needed.
  AllocaInst *
  createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                           ArrayRef<Instruction *> StackRestorePoints,
                           Value *StaticTop, bool NeedDynamicTop,
                           Value *UnsafeStackPtr, StringRef StackID);


  /// Replace all allocas in \p DynamicAllocas with code to allocate
  /// space dynamically on the unsafe stack and store the dynamic unsafe stack
  /// top to \p DynamicTop if non-null.
  void moveDynamicAllocasToHeap(Function &F, ArrayRef<AllocaInst *> DynamicAllocas);

  bool IsSafeStackAlloca(const Value *AllocaPtr, uint64_t AllocaSize);

  bool IsMemIntrinsicSafe(const MemIntrinsic *MI, const Use &U,
                          const Value *AllocaPtr, uint64_t AllocaSize);
  bool IsAccessSafe(Value *Addr, uint64_t Size, const Value *AllocaPtr,
                    uint64_t AllocaSize);

  // Checks
  bool ChecksOnFunc(Function &F, ObjectSizeOffsetVisitor &ObjSizeVis);
  std::tuple<Value *, Value *> InsertCheck(Instruction &I, Value &addr, bool write, Type* ptrType);
  std::tuple<Value *, Value *> InsertCheckMeta(Instruction &I, Value &addr, bool write, Type* ptrType, Value *EndOfObj, Value *Tag, bool slowZero);
  void InsertCheckRange(Instruction &I, Value *start, Value *end, Type* ptrType);

  Value *getOrCreateFnShadowBase();
  void AccumulateToUnsafeStackAlloca(Value *V, SmallPtrSetImpl<Value*> &Visited, Value **found, Constant **og_size);
  Constant* getUnsafeStackObjOgSize(Value *V);

  bool isSafeAccess(ObjectSizeOffsetVisitor &ObjSizeVis, Value *Addr, uint64_t TypeSize);
  bool isSafeAccessBoost(ObjectSizeOffsetVisitor &ObjSizeVis, Instruction *IndexInst, Value *Addr, Function *F) const;
  void OptStaticallySafeAccesses(ObjectSizeOffsetVisitor &ObjSizeVis, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument);
  void OptStaticallySafeAllocas(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument);
  void OptStaticallySafeUnsafeStack(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument);
  void sequentialExecuteOptimizationBoost(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument);
  void baseAddrOffsetMapPreprocessing(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi);

  enum addrType loopOptimizationCategorise(Function &F, Loop *L, InterestingMemoryOperand Oper, ScalarEvolution *SE);

  void sequentialExecuteOptimizationPostDom(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA);
  void sequentialExecuteOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA);
  bool ConservativeCallIntrinsicCheck(Instruction *InstStart, Instruction *InstEnd, std::set<Instruction *> &callIntrinsicSet, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT);
  bool isPostDominatWrapper(Instruction *InstStart, Instruction *TargetInst, llvm::PostDominatorTree &PDT);
  void ConservativeCallIntrinsicCollect(Function &F, std::set<Instruction *> &callIntrinsicSet);

  void ExploreLoopConditions(BasicBlock *BB, Loop *L, DenseMap<BasicBlock *, SmallVector<InvPathConditions, 8>> &AccessCondPaths, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, std::set<Value *> &TCond, std::set<Value *> &FCond, std::set<std::pair<Value *, Value *>> &SCond, LoopInfo *LI, ScalarEvolution *SE);

  // void WalkInvCondPaths(Function &F, DenseMap<BasicBlock *, SmallVector<InvPathConditions, 8>> &AccessCondPaths, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, LoopInfo *LI, ScalarEvolution *SE);

  void ClassifyLoopAccesses(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, 
  SmallVector<InterestingMemoryOperand, 16> &invariants, LoopInfo *LI, ScalarEvolution *SE);

  bool LoopPtrReuseMetadata(Function &F, LoopRangeCheck &Check, SmallVector<CheckMergeCands, 16> &CheckMergers, SmallVector<CheckMergeCands, 16> &FuncMetaReuse, std::set<Instruction *> &optimized);
  
  bool PtrContainsCallUse(Value *Ptr, Loop *L);

  void LoopOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<InterestingMemoryOperand, 16> &invariants, SmallVector<CheckMergeCands, 16> &CheckMergers, SmallVector<CheckMergeCands, 16> &FuncMetaReuse, LoopInfo *LI, ScalarEvolution *SE);


  void MetadataSharingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &FuncMetaMergers, AliasAnalysis *AA, ScalarEvolution *SE);

  bool MoveAddrUp(Instruction *InsertPt, Instruction *Addr/*, PostDominatorTree &PDT*/);
  bool IsAvailableToMove(Instruction *Ptr, Instruction *InsertPt, SmallPtrSetImpl<Value*> &Visited);

  void BlockSharingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &CheckMergers, AliasAnalysis *AA, ScalarEvolution *SE);

  void MinMaxMergingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<MinMaxPair, 16> &NegPosPairs, SmallVector<MinMaxChain, 16> &MinMaxChains, AliasAnalysis *AA, ScalarEvolution *SE);

  void NegPosMergingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<NegPosPair, 32> &MinMaxPairs, AliasAnalysis *AA, ScalarEvolution *SE);

  void FindFunctionMetaSharingCandidates(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &CheckMergers, AliasAnalysis *AA, ScalarEvolution *SE);

  void FindMinMaxPairs(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<MinMaxPair, 16> &MinMaxPairs, SmallVector<MinMaxChain, 16> &MinMaxChains, AliasAnalysis *AA, ScalarEvolution *SE);

  void FindNegPosPairs(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<NegPosPair, 32> &NegPosPairs, AliasAnalysis *AA, ScalarEvolution *SE);

  void FindBlockCheckSharingCandidates(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &CheckMergers, AliasAnalysis *AA, ScalarEvolution *SE);

  void getInterestingMemoryOperands(Instruction &I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);

  void InstrumentCalls(Function &F);

public:

  SafeStack(Function &F, const TargetLoweringBase &TL, const DataLayout &DL,
            DomTreeUpdater *DTU, ScalarEvolution &SE, DominatorTree &DT, DominanceFrontier &DF,
            SizedStackRuntime &RT, TargetLibraryInfo &TLI, LoopInfo &LI, AliasAnalysis *AA/*, DependenceInfo *DI*/)
      : F(F), TL(TL), DL(DL), DTU(DTU), SE(SE), DT(DT), DF(DF), RT(RT), TLI(TLI), LI(LI), AA(AA), /*DI(DI),*/ 
        StackPtrTy(Type::getInt8PtrTy(F.getContext())),
        IntPtrTy(DL.getIntPtrType(F.getContext())),
        Int32Ty(Type::getInt32Ty(F.getContext())),
        Int8Ty(Type::getInt8Ty(F.getContext())) {
          
          FunctionType *swiftsan_report_ty = FunctionType::get(Type::getVoidTy(F.getContext()), {Type::getInt8PtrTy(F.getContext()), Type::getInt8PtrTy(F.getContext()), Type::getInt8PtrTy(F.getContext())}, false);
          swiftsan_report_fn = F.getParent()->getOrInsertFunction("__subsan_report", swiftsan_report_ty);

          FunctionType *swiftsan_argv_ty = FunctionType::get(Type::getInt8PtrTy(F.getContext()), {Type::getInt32Ty(F.getContext()), Type::getInt8PtrTy(F.getContext())}, false);
          swiftsan_argv_fn = F.getParent()->getOrInsertFunction("__subsan_move_argv_to_heap", swiftsan_argv_ty);

          // mem
          LLVMContext &C = F.getParent()->getContext();
          IRBuilder<> builder(C);
          SwiftsanMemmove = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memmove", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);
          SwiftsanMemcpy = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memcpy", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);
          SwiftsanMemset = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memset", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt32Ty(), IntPtrTy);

          SwiftsanMemmoveSrc = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memmove_src_only", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);
          SwiftsanMemcpySrc = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memcpy_src_only", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);

          SwiftsanMemmoveDst = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memmove_dst_only", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);
          SwiftsanMemcpyDst = F.getParent()->getOrInsertFunction(
              interceptPrefix + "memcpy_dst_only", builder.getInt8PtrTy(),
              builder.getInt8PtrTy(), builder.getInt8PtrTy(), IntPtrTy);

  }

  // Run the transformation on the associated function.
  // Returns whether the function was changed.
  bool run();
};

constexpr Align SafeStack::StackAlignment;

uint64_t SafeStack::getStaticAllocaAllocationSize(const AllocaInst* AI) {
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

bool SafeStack::IsAccessSafe(Value *Addr, uint64_t AccessSize,
                             const Value *AllocaPtr, uint64_t AllocaSize) {
  const SCEV *AddrExpr = SE.getSCEV(Addr);
  const auto *Base = dyn_cast<SCEVUnknown>(SE.getPointerBase(AddrExpr));
  if (!Base || Base->getValue() != AllocaPtr) {
    LLVM_DEBUG(
        dbgs() << "[SafeStack] "
               << (isa<AllocaInst>(AllocaPtr) ? "Alloca " : "ByValArgument ")
               << *AllocaPtr << "\n"
               << "SCEV " << *AddrExpr << " not directly based on alloca\n");
    return false;
  }

  const SCEV *Expr = SE.removePointerBase(AddrExpr);
  uint64_t BitWidth = SE.getTypeSizeInBits(Expr->getType());
  ConstantRange AccessStartRange = SE.getUnsignedRange(Expr);
  ConstantRange SizeRange =
      ConstantRange(APInt(BitWidth, 0), APInt(BitWidth, AccessSize));
  ConstantRange AccessRange = AccessStartRange.add(SizeRange);
  ConstantRange AllocaRange =
      ConstantRange(APInt(BitWidth, 0), APInt(BitWidth, AllocaSize));
  bool Safe = AllocaRange.contains(AccessRange);

  LLVM_DEBUG(
      dbgs() << "[SafeStack] "
             << (isa<AllocaInst>(AllocaPtr) ? "Alloca " : "ByValArgument ")
             << *AllocaPtr << "\n"
             << "            Access " << *Addr << "\n"
             << "            SCEV " << *Expr
             << " U: " << SE.getUnsignedRange(Expr)
             << ", S: " << SE.getSignedRange(Expr) << "\n"
             << "            Range " << AccessRange << "\n"
             << "            AllocaRange " << AllocaRange << "\n"
             << "            " << (Safe ? "safe" : "unsafe") << "\n");

  return Safe;
}

bool SafeStack::IsMemIntrinsicSafe(const MemIntrinsic *MI, const Use &U,
                                   const Value *AllocaPtr,
                                   uint64_t AllocaSize) {

  if (auto MTI = dyn_cast<MemTransferInst>(MI)) {
    if (MTI->getRawSource() != U && MTI->getRawDest() != U)
      return true;
  } else {
    if (MI->getRawDest() != U)
      return true;
  }

  const auto *Len = dyn_cast<ConstantInt>(MI->getLength());
  // Non-constant size => unsafe. FIXME: try SCEV getRange.
  if (!Len) return false;
  return IsAccessSafe(U, Len->getZExtValue(), AllocaPtr, AllocaSize);
}

/// Check whether a given allocation must be put on the safe
/// stack or not. The function analyzes all uses of AI and checks whether it is
/// only accessed in a memory safe way (as decided statically).
bool SafeStack::IsSafeStackAlloca(const Value *AllocaPtr, uint64_t AllocaSize) {
  // Go through all uses of this alloca and check whether all accesses to the
  // allocated object are statically known to be memory safe and, hence, the
  // object can be placed on the safe stack.
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 8> WorkList;
  WorkList.push_back(AllocaPtr);

  // A DFS search through all uses of the alloca in bitcasts/PHI/GEPs/etc.
  while (!WorkList.empty()) {
    const Value *V = WorkList.pop_back_val();
    for (const Use &UI : V->uses()) {
      auto I = cast<Instruction>(UI.getUser());
      assert(V == UI.get());

      switch (I->getOpcode()) {
      case Instruction::Load:{
        if (!IsAccessSafe(UI, DL.getTypeStoreSize(I->getType()), AllocaPtr,
                          AllocaSize))
            //&& !isSafeAccessBoost(ObjSizeVis, I, UI, I->getFunction())
            //&& !isSafeAccess(ObjSizeVis, UI, DL.getTypeStoreSize(I->getType())))
          return false;
        else
          SafeMemOps.push_back(I);
        break;
      }

      case Instruction::VAArg:
        // "va-arg" from a pointer is safe.
        break;
      case Instruction::Store: {
        if (V == I->getOperand(0)) {
          // Stored the pointer - conservatively assume it may be unsafe.
          LLVM_DEBUG(dbgs()
                     << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                     << "\n            store of address: " << *I << "\n");
          return false;
        }

        if (!IsAccessSafe(UI, DL.getTypeStoreSize(I->getOperand(0)->getType()),
                          AllocaPtr, AllocaSize))
          //&& !isSafeAccessBoost(ObjSizeVis, I, UI, I->getFunction())
          //&& !isSafeAccess(ObjSizeVis, UI, DL.getTypeStoreSize(I->getOperand(0)->getType())))
          return false;
        else
          SafeMemOps.push_back(I);
        break;
      }
      case Instruction::Ret:
        // Information leak.
        return false;

      case Instruction::Call:
      case Instruction::Invoke: {
        const CallBase &CS = *cast<CallBase>(I);

        if (I->isLifetimeStartOrEnd())
          continue;

        // UI is a use of the Alloca
        // I is a user of the UI use
        if (const MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I)) {
          if (!IsMemIntrinsicSafe(MI, UI, AllocaPtr, AllocaSize)) {
            LLVM_DEBUG(dbgs()
                       << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                       << "\n            unsafe memintrinsic: " << *I << "\n");
            return false;
          }
          continue;
        }

        // LLVM 'nocapture' attribute is only set for arguments whose address
        // is not stored, passed around, or used in any other non-trivial way.
        // We assume that passing a pointer to an object as a 'nocapture
        // readnone' argument is safe.
        // FIXME: a more precise solution would require an interprocedural
        // analysis here, which would look at all uses of an argument inside
        // the function being called.
        auto B = CS.arg_begin(), E = CS.arg_end();
        for (const auto *A = B; A != E; ++A)
          if (A->get() == V)
            if (!(CS.doesNotCapture(A - B) && (CS.doesNotAccessMemory(A - B) ||
                                               CS.doesNotAccessMemory()))) {
              LLVM_DEBUG(dbgs() << "[SafeStack] Unsafe alloca: " << *AllocaPtr
                                << "\n            unsafe call: " << *I << "\n");
              return false;
            }
        continue;
      }

      default:
        if (Visited.insert(I).second)
          WorkList.push_back(cast<const Instruction>(I));
      }
    }
  }

  // All uses of the alloca are safe, we can place it on the safe stack.
  return true;
}

void SafeStack::findInsts(Function &F,
                          SmallVectorImpl<AllocaInst *> &StaticAllocas,
                          SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                          SmallVectorImpl<Argument *> &ByValArguments,
                          SmallVectorImpl<Instruction *> &Returns,
                          SmallVectorImpl<Instruction *> &StackRestorePoints) {
  for (Instruction &I : instructions(&F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
      ++NumAllocas;

      uint64_t Size = getStaticAllocaAllocationSize(AI);
      if (IsSafeStackAlloca(AI, Size)){
        continue;
      }

      if (AI->isStaticAlloca()) {
        ++NumUnsafeStaticAllocas;
        StaticAllocas.push_back(AI);
      } else {
        ++NumUnsafeDynamicAllocas;
        DynamicAllocas.push_back(AI);
      }
    } else if (auto RI = dyn_cast<ReturnInst>(&I)) {
      if (CallInst *CI = I.getParent()->getTerminatingMustTailCall())
        Returns.push_back(CI);
      else
        Returns.push_back(RI);
    } else if (auto CI = dyn_cast<CallInst>(&I)) {
      // setjmps require stack restore.
      if (CI->getCalledFunction() && CI->canReturnTwice())
        StackRestorePoints.push_back(CI);
    } else if (auto LP = dyn_cast<LandingPadInst>(&I)) {
      // Exception landing pads require stack restore.
      StackRestorePoints.push_back(LP);
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::gcroot)
        report_fatal_error(
            "gcroot intrinsic not compatible with safestack attribute");
    }
  }
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    uint64_t Size = DL.getTypeStoreSize(Arg.getParamByValType());
    if (IsSafeStackAlloca(&Arg, Size))
      continue;
    ++NumUnsafeByValArguments;
    ByValArguments.push_back(&Arg);
  }
}

AllocaInst *
SafeStack::createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                                    ArrayRef<Instruction *> StackRestorePoints,
                                    Value *StaticTop, bool NeedDynamicTop,
                                    Value *UnsafeStackPtr, StringRef StackID) {
  assert(StaticTop && "The stack top isn't set.");

  if (StackRestorePoints.empty())
    return nullptr;

  // We need the current value of the shadow stack pointer to restore
  // after longjmp or exception catching.

  // FIXME: On some platforms this could be handled by the longjmp/exception
  // runtime itself.

  AllocaInst *DynamicTop = nullptr;
  if (NeedDynamicTop) {
    // If we also have dynamic alloca's, the stack pointer value changes
    // throughout the function. For now we store it in an alloca.
    DynamicTop = IRB.CreateAlloca(StackPtrTy, /*ArraySize=*/nullptr,
                                  "unsafe_stack_dynamic_ptr_" + StackID);
    Instruction *IStore = IRB.CreateStore(StaticTop, DynamicTop);
    IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
  }

  // Restore current stack pointer after longjmp/exception catch.
  for (Instruction *I : StackRestorePoints) {
    ++NumUnsafeStackRestorePoints;

    IRB.SetInsertPoint(I->getNextNode());
    Value *CurrentTop;
    if (DynamicTop) {
      Instruction *ILoad = IRB.CreateLoad(StackPtrTy, DynamicTop);
      ILoad->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
      CurrentTop = ILoad;
    } else {
      CurrentTop = StaticTop;
    }
    Instruction *IStore = IRB.CreateStore(CurrentTop, UnsafeStackPtr);
    IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    size_t sid = RT.getTypeIndex(StackID);
    RestorePoints[I].push_back(sid);
  }

  return DynamicTop;
}

/// We explicitly compute and set the unsafe stack layout for all unsafe
/// static alloca instructions. We save the unsafe "base pointer" in the
/// prologue into a local variable and restore it in the epilogue.
Value *SafeStack::moveStaticAllocasToUnsafeStack(
    IRBuilder<> &IRB, Function &F, ArrayRef<AllocaInst *> StaticAllocas,
    ArrayRef<Argument *> ByValArguments, Instruction *BasePointer,
    AllocaInst *StackGuardSlot, Value *UnsafeStackPtr, StringRef StackID,
    SmallVectorImpl<std::tuple<Value*, uint64_t, uint8_t>> &TaggedStackAllocas) {
  if (StaticAllocas.empty() && ByValArguments.empty())
    return BasePointer;

  DIBuilder DIB(*F.getParent());

  StackLifetime SSC(F, StaticAllocas, StackLifetime::LivenessType::May);
  static const StackLifetime::LiveRange NoColoringRange(1, true);
  if (ClColoring)
    SSC.run();

  for (const auto *I : SSC.getMarkers()) {
    auto *Op = dyn_cast<Instruction>(I->getOperand(1));
    const_cast<IntrinsicInst *>(I)->eraseFromParent();
    // Remove the operand bitcast, too, if it has no more uses left.
    if (Op && Op->use_empty())
      Op->eraseFromParent();
  }

  // Unsafe stack always grows down.
  StackLayout SSL(StackAlignment);
  if (StackGuardSlot) {
    Type *Ty = StackGuardSlot->getAllocatedType();
    Align Align = std::max(DL.getPrefTypeAlign(Ty), StackGuardSlot->getAlign());
    SSL.addObject(StackGuardSlot, getStaticAllocaAllocationSize(StackGuardSlot),
                  Align, SSC.getFullLiveRange());
  }

  // All entries have the same size on the stack.
  // FIXME: getting this from the string is dirty
  uint64_t SizeClass = std::stoll(StackID.substr(StackID.rfind('_') + 1).str());
  (void)SizeClass;
  Type* Int64Ty = Type::getInt64Ty(F.getContext());
  Type* I8Ty = Type::getInt8Ty(F.getContext());

  auto NextStackTag = []() -> uint8_t {
    static uint8_t module_alloca_tag = subsanStackTagLo;
    uint8_t t = module_alloca_tag;
    // stack range cycle (per-arch — TBI 1..85, LAM 1..21).
    module_alloca_tag = (module_alloca_tag >= subsanStackTagHi)
                            ? subsanStackTagLo
                            : (module_alloca_tag + 1);
    return t;
  };
  FunctionCallee TagMemoryFn = F.getParent()->getOrInsertFunction(
      "__subsan_tag_memory", Int64Ty, Int64Ty, Int64Ty, I8Ty);


  for (Argument *Arg : ByValArguments) {
    Type *Ty = Arg->getParamByValType();
    uint64_t Size = DL.getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    Align Align = DL.getPrefTypeAlign(Ty);
    if (auto A = Arg->getParamAlign())
      Align = std::max(Align, *A);

      // Alignment must be compatible with size class.
    if (SizeClass % Align.value()) {
      FAIL("alignment " << Align.value() << " does not divide size class " << SizeClass << ":\n" << *Arg);
    }

    SSL.addObject(Arg, SizeClass, Align, SSC.getFullLiveRange());
  }

  for (AllocaInst *AI : StaticAllocas) {
    Type *Ty = AI->getAllocatedType();
    uint64_t Size = getStaticAllocaAllocationSize(AI);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    Align Align = std::max(DL.getPrefTypeAlign(Ty), AI->getAlign());

    // Alignment must be compatible with size class.
    if (SizeClass % Align.value()) {
      FAIL("alignment " << Align.value() << " does not divide size class " << SizeClass << ":\n" << *AI);
    }

    SSL.addObject(AI, SizeClass, Align, ClColoring ? SSC.getLiveRange(AI) : NoColoringRange);
  }

  SSL.computeLayout();
  Align FrameAlignment = SSL.getFrameAlignment();

  // FIXME: tell SSL that we start at a less-then-MaxAlignment aligned location
  // (AlignmentSkew).
  if (FrameAlignment > StackAlignment) {
      FAIL("frame alignment " << FrameAlignment.value() << " bigger than stack alignment " <<
               RT.StackAlignment << ", need to realign but this would mess with sizeclass offsets");
    // Re-align the base pointer according to the max requested alignment.
    // IRB.SetInsertPoint(BasePointer->getNextNode());
    // BasePointer = cast<Instruction>(IRB.CreateIntToPtr(
    //     IRB.CreateAnd(
    //         IRB.CreatePtrToInt(BasePointer, IntPtrTy),
    //         ConstantInt::get(IntPtrTy, ~(FrameAlignment.value() - 1))),
    //     StackPtrTy));
  }

  if (SizeClass % FrameAlignment.value())
    FAIL("size class " << SizeClass << " is not a multiple of frame alignment " << FrameAlignment.value());

  IRB.SetInsertPoint(BasePointer->getNextNode());

  if (StackGuardSlot) {
    unsigned Offset = SSL.getObjectOffset(StackGuardSlot);
    Value *Off = IRB.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -Offset));
    Value *NewAI =
        IRB.CreateBitCast(Off, StackGuardSlot->getType(), "StackGuardSlot");

    // Replace alloc with the new location.
    StackGuardSlot->replaceAllUsesWith(NewAI);
    StackGuardSlot->eraseFromParent();
  }

  SetVector<uint64_t> ObjectOffsets;

  for (Argument *Arg : ByValArguments) {
    unsigned Offset = SSL.getObjectOffset(Arg);
    if (Offset % SizeClass)
      FAIL("object offset " << Offset << " is not a multiple of sizeclass " << SizeClass);

    ObjectOffsets.insert(Offset);
    MaybeAlign Align(SSL.getObjectAlignment(Arg));
    Type *Ty = Arg->getParamByValType();

    uint64_t Size = DL.getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    Value *Off = IRB.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -Offset));
    Value *NewArg = IRB.CreateBitCast(Off, Arg->getType(),
                                     Arg->getName() + ".unsafe-byval");

    // Replace alloc with the new location.
    replaceDbgDeclare(Arg, BasePointer, DIB, DIExpression::ApplyOffset,
                      -Offset);
    IRB.SetInsertPoint(cast<Instruction>(NewArg)->getNextNode());

#ifdef ALLOCA_TAGGING
  if(!isImplicitTagging) { // alloca tagging is not needed when implicit
    uint8_t this_tag = NextStackTag();
    uint64_t tag_shifted = uint64_t(this_tag) << subsanTagShift;
    Value *AllocaInt = IRB.CreatePtrToInt(NewArg, IntPtrTy);
    Value *TaggedAlloca = IRB.CreateOr(AllocaInt, tag_shifted);
    NewArg = IRB.CreateIntToPtr(TaggedAlloca, Arg->getType());

    uint64_t AlignedSize = (Size + 15) & ~15ULL;
    if (ClSubsanInlineTagMemory) {
      emitInlineTagMemory(IRB, F.getParent(), AllocaInt, Size, this_tag);
    } else {
      IRB.CreateCall(TagMemoryFn,
          {AllocaInt, ConstantInt::get(Int64Ty, Size),
           ConstantInt::get(I8Ty, this_tag)});
    }
    TaggedStackAllocas.push_back(std::make_tuple(AllocaInt, Size, this_tag));
    (void)AlignedSize;
  }
#endif

    Arg->replaceAllUsesWith(NewArg);
    Instruction *IMemcpy = IRB.CreateMemCpy(Off, Align, Arg, Arg->getParamAlign(), Size);
    IMemcpy->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    // Value *MetaOffset = IRB.CreateGEP(Int8Ty, NewArg, ConstantInt::get(Int64Ty, -8));
    // Value *MetaPtr = IRB.CreateBitCast(MetaOffset, Type::getInt64PtrTy(F.getContext()));
    // Value *ArgPtrInt = IRB.CreatePtrToInt(NewArg, IntPtrTy);
    // Value *EndOfStackObj = IRB.CreateAdd(ArgPtrInt, ConstantInt::get(Int64Ty, Size));
    // Instruction *IStore = IRB.CreateStore(EndOfStackObj, MetaPtr);
    // IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
  }

  // Allocate space for every unsafe static AllocaInst on the unsafe stack.
  for (AllocaInst *AI : StaticAllocas) {
    IRB.SetInsertPoint(AI);
    unsigned Offset = SSL.getObjectOffset(AI);
    if (Offset % SizeClass)
      FAIL("object offset " << Offset << " is not a multiple of sizeclass " << SizeClass);
    ObjectOffsets.insert(Offset);

    replaceDbgDeclare(AI, BasePointer, DIB, DIExpression::ApplyOffset, -Offset);
    replaceDbgValueForAlloca(AI, BasePointer, DIB, -Offset);

    // Replace uses of the alloca with the new location.
    // Insert address calculation close to each use to work around PR27844.
    std::string Name = std::string(AI->getName()) + ".unsafe";
  if(!isImplicitTagging) {
    // by default SafeStack places calculation of new alloca address close to the use to avoid
    // using registers for a long time.
    // however if we also need to perform ptr tag arithmetic it may make more sense to not
    // recalculate at every use site.
#ifndef ALLOCA_TAGGING
    IRB.SetInsertPoint(BasePointer->getNextNode());
#endif
    Value *Off = IRB.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                                     ConstantInt::get(Int32Ty, -Offset));
    Value *Replacement = IRB.CreateBitCast(Off, AI->getType(), Name);
#ifdef ALLOCA_TAGGING
    uint8_t this_tag = NextStackTag();
    uint64_t tag_shifted = uint64_t(this_tag) << subsanTagShift;
    Value *AllocaInt = IRB.CreatePtrToInt(Replacement, IntPtrTy);
    Value *TaggedAlloca = IRB.CreateOr(AllocaInt, tag_shifted);
    Replacement = IRB.CreateIntToPtr(TaggedAlloca, AI->getType());

    uint64_t original_size = getStaticAllocaAllocationSize(AI);
    if (ClSubsanInlineTagMemory) {
      emitInlineTagMemory(IRB, F.getParent(), AllocaInt, original_size, this_tag);
    } else {
      IRB.CreateCall(TagMemoryFn,
          {AllocaInt, ConstantInt::get(Int64Ty, original_size),
           ConstantInt::get(I8Ty, this_tag)});
    }
    TaggedStackAllocas.push_back(std::make_tuple(AllocaInt, original_size, this_tag));

    // IRB.SetInsertPoint(AI->getNextNode());
    // Value *MetaOffset = IRB.CreateGEP(Int8Ty, Replacement, ConstantInt::get(Int64Ty, -8));
    // Value *MetaPtr = IRB.CreateBitCast(MetaOffset, Type::getInt64PtrTy(F.getContext()));
    // Value *EndOfStackObj = IRB.CreateAdd(TaggedAlloca, ConstantInt::get(Int64Ty, original_size));
    // Instruction *IStore = IRB.CreateStore(EndOfStackObj, MetaPtr);
    // IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
#endif

    if(BinaryOperator *binOR = dyn_cast<BinaryOperator>(TaggedAlloca)){
      MDNode* temp_N = MDNode::get(F.getContext(), ConstantAsMetadata::get(ConstantInt::get(Int64Ty, original_size)));
      MDNode* N = MDNode::get(F.getContext(), temp_N);
      binOR->setMetadata("unsafe.stack.taggedor.og_size", N);
    }

    AI->replaceAllUsesWith(Replacement);
  }
  else{ // implicit tagging: replace at use-site directly, no need to tag so its cheap
    uint64_t original_size = getStaticAllocaAllocationSize(AI);
    while (!AI->use_empty()) {
      Use &U = *AI->use_begin();
      Instruction *User = cast<Instruction>(U.getUser());

      Instruction *InsertBefore;
      if (auto *PHI = dyn_cast<PHINode>(User))
        InsertBefore = PHI->getIncomingBlock(U)->getTerminator();
      else
        InsertBefore = User;

      IRBuilder<> IRBUser(InsertBefore);
      Value *Off = IRBUser.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                                     ConstantInt::get(Int32Ty, -Offset));
      Value *Replacement = IRBUser.CreateBitCast(Off, AI->getType(), Name);

      if(GetElementPtrInst* gepp = dyn_cast<GetElementPtrInst>(Off)){
        MDNode* temp_N = MDNode::get(F.getContext(), ConstantAsMetadata::get(ConstantInt::get(Int64Ty, original_size)));
        MDNode* N = MDNode::get(F.getContext(), temp_N);
        gepp->setMetadata("unsafe.stack.og_size", N);
      }

      if (auto *PHI = dyn_cast<PHINode>(User))
        // PHI nodes may have multiple incoming edges from the same BB (why??),
        // all must be updated at once with the same incoming value.
        PHI->setIncomingValueForBlock(PHI->getIncomingBlock(U), Replacement);
      else
        U.set(Replacement);
    }

  }

    AI->eraseFromParent();
  }

  // Re-align BasePointer so that our callees would see it aligned as
  // expected.
  // FIXME: no need to update BasePointer in leaf functions.
  unsigned FrameSize = alignTo(SSL.getFrameSize(), StackAlignment);

  // The frame size must be a multiple of the size class to make the redzone
  // positions predictible, which is required for the slow path and for redzone
  // initialization on page fault.
  assert(SSL.getFrameSize() % SizeClass == 0);

  MDBuilder MDB(F.getContext());
  SmallVector<Metadata *, 2> Data;
  Data.push_back(MDB.createString("unsafe-stack-size"));
  Data.push_back(MDB.createConstant(ConstantInt::get(Int32Ty, FrameSize)));
  MDNode *MD = MDTuple::get(F.getContext(), Data);
  F.setMetadata(LLVMContext::MD_annotation, MD);

  // Update shadow stack pointer in the function epilogue.
  IRB.SetInsertPoint(BasePointer->getNextNode());

  Value *StaticTop =
      IRB.CreateGEP(Int8Ty, BasePointer, ConstantInt::get(Int32Ty, -FrameSize),
                    "unsafe_stack_static_top_" + StackID);
  Instruction *IStore = IRB.CreateStore(StaticTop, UnsafeStackPtr);
  IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

  return StaticTop;
}

static void findReachableExits(BasicBlock *BB,
    SmallVectorImpl<Instruction*> &Exits,
    SmallPtrSetImpl<BasicBlock*> &Visited) {
  if (Visited.count(BB))
    return;
  Visited.insert(BB);

  if (ReturnInst *RI  = dyn_cast<ReturnInst>(BB->getTerminator())) {
    Exits.push_back(RI);
  } else {
    for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
      findReachableExits(*SI, Exits, Visited);
  }
}

template<unsigned N = 4>
static SmallVector<Instruction*, N> findReachableExits(Instruction *I) {
  SmallVector<Instruction*, N> Exits;
  SmallPtrSet<BasicBlock*, 8> Visited;
  findReachableExits(I->getParent(), Exits, Visited);
  return Exits;
}

static bool findDominancePathBetween(BasicBlock *A, BasicBlock *B,
                                     SmallVectorImpl<BasicBlock*> &Path,
                                     DominatorTree &DT, DominanceFrontier &DF) {
  if (DT.dominates(A, B))
    return true;

  for (BasicBlock *FB : DF.calculate(DT, DT.getNode(A))) {
    if (FB != A) {
      Path.push_back(FB);
      if (findDominancePathBetween(FB, B, Path, DT, DF))
        return true;
#if LLVM_VERSION_MAJOR > 7
      Path.pop_back();
#else
      Path.pop_back_val();
#endif
    }
  }

  return false;
}

template<unsigned N = 2>
static SmallVector<BasicBlock*, N> findDominancePathBetween(
                                      BasicBlock *A, BasicBlock *B,
                                      DominatorTree &DT, DominanceFrontier &DF) {
  SmallVector<BasicBlock*, N> Path;
  bool Success = findDominancePathBetween(A, B, Path, DT, DF);
  assert(Success);
  return Path;
}

static Instruction *makeDominating(Instruction *Alloc, Instruction *RI,
                                   DominatorTree &DT, DominanceFrontier &DF) {
  PointerType *Ty = cast<PointerType>(Alloc->getType());
  Instruction *Prev = Alloc;

  for (BasicBlock *BB : findDominancePathBetween(Alloc->getParent(), RI->getParent(), DT, DF)) {
    PHINode *PN = PHINode::Create(Ty, 2, Alloc->getName() + ".prop", BB->getFirstNonPHI());
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *Pred = *PI;
      if (DT.dominates(Prev->getParent(), Pred))
        PN->addIncoming(Prev, Pred);
      else
        PN->addIncoming(ConstantPointerNull::get(Ty), Pred);
    }
    Prev = PN;
  }

  assert(DT.dominates(Prev, RI));
  return Prev;
}

void SafeStack::moveDynamicAllocasToHeap(Function &F, ArrayRef<AllocaInst*> DynamicAllocas) {
  DIBuilder DIB(*F.getParent());
  IRBuilder<> IRB(F.getContext());

  for (AllocaInst *AI : DynamicAllocas) {
    LLVM_DEBUG(dbgs() << "[SizedStack] Move dynamic alloca to heap:" << *AI << "\n");
    IRB.SetInsertPoint(AI);

    // Compute allocation size in bytes
    Value *Size = AI->getArraySize();
    if (Size->getType() != IntPtrTy)
      Size = IRB.CreateIntCast(Size, IntPtrTy, false);
    uint64_t TySize = DL.getTypeAllocSize(AI->getAllocatedType());
    if (TySize != 1)
      Size = IRB.CreateMul(Size, ConstantInt::get(IntPtrTy, TySize));

    // Replace alloca with call to heap allocation helper in runtime
    Value *Align = ConstantInt::get(IntPtrTy, AI->getAlign().value());
    Instruction *HeapAlloc = IRB.CreateCall(RT.DynAllocFunc, {Size, Align});
    HeapAlloc->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    Value *Replacement = HeapAlloc;
    if (Replacement->getType() != AI->getType())
      Replacement = IRB.CreatePointerCast(Replacement, AI->getType());
    Replacement->takeName(AI);
    replaceDbgDeclare(AI, Replacement, DIB, DIExpression::ApplyOffset, 0);

    AI->replaceAllUsesWith(Replacement);
    AI->eraseFromParent();

    // We always want to free the allocation if the return is dominated by the
    // allocation. If the return is not dominated, but is reachable, we insert a
    // PHI(alloc, NULL) node at the dominance frontier leading to the return,
    // and pass that to the free function (which must do a NULL check that
    // should be optimized away for non-phi pointers).
    for (Instruction *RI : findReachableExits(HeapAlloc)) {
      Instruction *FreedPointer = makeDominating(HeapAlloc, RI, DT, DF);
      Function *Helper = FreedPointer == HeapAlloc ? RT.DynFreeFunc : RT.DynFreeOptFunc;
      IRB.SetInsertPoint(RI);
      Instruction *ICall = IRB.CreateCall(Helper, {FreedPointer});
      ICall->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
    }

    ++NumDynAllocasMovedToHeap;
  }
}

Constant* SafeStack::getUnsafeStackObjOgSize(Value *V){
  if(BinaryOperator *bo = dyn_cast<BinaryOperator>(V)){
    // OR instruction for pointer tagging with TBI
    if(bo->getOpcode() == Instruction::Or){
        // if the OR has this metadata, it has to be from an unsafe stack alloca initial tagging
        if (MDNode* N = bo->getMetadata("unsafe.stack.taggedor.og_size")) {
          Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
          return val;
        }
    }
  }
  else if(GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(V)){
    // UnsafeStack accesses are always through a GEP
    if(LoadInst* load = dyn_cast<LoadInst>(gep->getPointerOperand())){
      if(load->getPointerOperand() == load->getModule()->getNamedValue(kUnsafeStackPtrVarTemp)){
        if (MDNode* N = gep->getMetadata("unsafe.stack.og_size")) {
          Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
          return val;
        }
      }
      else if (llvm::ConstantExpr *cexpr = dyn_cast<llvm::ConstantExpr>(load->getPointerOperand())) {
        llvm::Instruction *cexpr_as_inst = cexpr->getAsInstruction();
        if(GetElementPtrInst* in_load_gep = dyn_cast<GetElementPtrInst>(cexpr_as_inst)){
          if(in_load_gep->getPointerOperand() == load->getModule()->getNamedValue(kUnsafeStackPtrVarTemp)){
            if (MDNode* N = gep->getMetadata("unsafe.stack.og_size")) {
              Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
              return val;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

void SafeStack::AccumulateToUnsafeStackAlloca(Value *V, SmallPtrSetImpl<Value*> &Visited, Value **found, Constant **og_size){
  V = V->stripPointerCastsAndAliases();
  if (Visited.count(V))
    return;

  if(GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(V)){
    Visited.insert(gep);

    // UnsafeStack accesses are always through a GEP
    /// TODO: merge these load and cexpr blocks into one "found" boolean
    if(LoadInst* load = dyn_cast<LoadInst>(gep->getPointerOperand())){
      if(load->getPointerOperand() == load->getModule()->getNamedValue(kUnsafeStackPtrVarTemp)){
        *found = gep; // the base is the result of the GEP on the unsafestack ptr
        if (MDNode* N = gep->getMetadata("unsafe.stack.og_size")) {
          Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
          if(val){
            *og_size = val;
          }
        }
        return;
      }
      else if (llvm::ConstantExpr *cexpr = dyn_cast<llvm::ConstantExpr>(load->getPointerOperand())) {
        llvm::Instruction *cexpr_as_inst = cexpr->getAsInstruction();
        if(GetElementPtrInst* in_load_gep = dyn_cast<GetElementPtrInst>(cexpr_as_inst)){
          if(in_load_gep->getPointerOperand() == load->getModule()->getNamedValue(kUnsafeStackPtrVarTemp)){
            *found = gep; // the base is the result of the GEP on the unsafestack ptr
            if (MDNode* N = gep->getMetadata("unsafe.stack.og_size")) {
              Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
              if(val){
                *og_size = val;
              }
            }
            return;
          }
        }
      }
    }
    return AccumulateToUnsafeStackAlloca(gep->getPointerOperand(), Visited, found, og_size);
  }
  else if(CastInst *cast = dyn_cast<CastInst>(V)){
    Visited.insert(cast);
    return AccumulateToUnsafeStackAlloca(cast->getOperand(0), Visited, found, og_size);
  }
  else if(BinaryOperator *bo = dyn_cast<BinaryOperator>(V)){
    // OR instruction for pointer tagging with TBI
    if(bo->getOpcode() == Instruction::Or){
        // if the OR has this metadata, it has to be from an unsafe stack alloca initial tagging
        if (MDNode* N = bo->getMetadata("unsafe.stack.taggedor.og_size")) {
            *found = bo; // the base is the result of the tagging OR on the unsafe alloca
            Constant* val = dyn_cast<ConstantAsMetadata>(dyn_cast<MDNode>(N->getOperand(0))->getOperand(0))->getValue();
            if(val){
              *og_size = val;
          }
        }
      Visited.insert(bo);
      return;
    }
  }
  else{
    Visited.insert(V);
  }
}

Value *SafeStack::getOrCreateFnShadowBase() {
  if (FnShadowBase) return FnShadowBase;

  Module *M = F.getParent();
  LLVMContext &C = F.getContext();
  IntegerType *Int64Ty = Type::getInt64Ty(C);

  BasicBlock &Entry = F.getEntryBlock();
  IRBuilder<> IRB(&*Entry.getFirstInsertionPt());

  if (ClSubsanFixedShadowBase) {
    Value *RawConst = ConstantInt::get(Int64Ty, ClSubsanShadowOffset);
    InlineAsm *MovAsm = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty}, false),
        /*asm=*/"", /*constraints=*/"=r,0",
        /*hasSideEffects=*/false, /*isAlignStack=*/false,
        InlineAsm::AD_ATT, /*canThrow=*/false);
    FnShadowBase = IRB.CreateCall(MovAsm, {RawConst},
                                   "subsan.fn.shadow_base.fixed");
    return FnShadowBase;
  }

  Constant *ShadowBaseC =
      M->getOrInsertGlobal("__hwasan_shadow_memory_dynamic_address", Int64Ty);
  GlobalVariable *ShadowBaseGV = dyn_cast<GlobalVariable>(ShadowBaseC);
  if (ShadowBaseGV && !ShadowBaseGV->hasInitializer())
    ShadowBaseGV->setLinkage(GlobalValue::ExternalLinkage);
  LoadInst *L = IRB.CreateAlignedLoad(Int64Ty, ShadowBaseGV, Align(8),
                                       "subsan.fn.shadow_base");
  L->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(C, std::nullopt));
  FnShadowBase = L;
  return FnShadowBase;
}

void SafeStack::InsertCheckRange(Instruction &I, Value *start, Value *end, Type* ptrType) {

  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);

  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  IntegerType *Int64Ty = Type::getInt64Ty(M->getContext());
  Type *Int64PtrTy = PointerType::get(Int64Ty, 0);

  builder.SetInsertPoint(&I);

  Value *Target = start;

  uint64_t tag_shift = isImplicitTagging ? 41 : subsanTagShift;
  Value *PtrAsInt = builder.CreatePtrToInt(Target, IntPtrTy);
  Value *Tag = nullptr;
  if (ClSubsanCcmpFastPath) {
    Tag = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, tag_shift),
                              "subsan.range.tag");
  }

  Value *GranuleIdx;
  if (isX86 && ClSubsanLamBmi2Bextr) {
    unsigned MaskLen = llvm::countr_one(subsanGranuleIdxMask);
    uint64_t Ctrl = ((uint64_t)MaskLen << 8) | 4ULL;
    InlineAsm *BextrAsm = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false),
        StringRef("bextrq $2, $1, $0"),
        StringRef("=r,r,r,~{flags}"),
        /*hasSideEffects=*/false,
        /*isAlignStack*/false,
        InlineAsm::AD_ATT,
        /*canThrow*/false);
    GranuleIdx = builder.CreateCall(BextrAsm,
        {PtrAsInt, ConstantInt::get(Int64Ty, Ctrl)},
        "subsan.range.granule_idx");
  } else {
    Value *Shifted = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, 4));
    GranuleIdx = builder.CreateAnd(
        Shifted, ConstantInt::get(Int64Ty, subsanGranuleIdxMask),
        "subsan.range.granule_idx");
  }
  Value *ShadowBase;
  if (ClSubsanHoistShadowBase) {
    ShadowBase = getOrCreateFnShadowBase();
  } else if (ClSubsanFixedShadowBase) {
    ShadowBase = ConstantInt::get(Int64Ty, ClSubsanShadowOffset);
  } else {
    Constant *ShadowBaseC =
        M->getOrInsertGlobal("__hwasan_shadow_memory_dynamic_address", Int64Ty);
    GlobalVariable *ShadowBaseGV = dyn_cast<GlobalVariable>(ShadowBaseC);
    if (ShadowBaseGV && !ShadowBaseGV->hasInitializer())
      ShadowBaseGV->setLinkage(GlobalValue::ExternalLinkage);
    ShadowBase = builder.CreateAlignedLoad(
        Int64Ty, ShadowBaseGV, Align(8), "subsan.range.shadow_base");
    cast<LoadInst>(ShadowBase)->setMetadata(
        LLVMContext::MD_invariant_load, MDNode::get(C, std::nullopt));
  }
  Value *MetaPtr;
  if (isX86 && ClSubsanLamBmi2Bextr) {
    Value *ShadowI64Base = builder.CreateIntToPtr(ShadowBase, Int64PtrTy);
    MetaPtr = builder.CreateInBoundsGEP(Int64Ty, ShadowI64Base, GranuleIdx,
                                          "subsan.range.meta_ptr");
  } else if (isX86) {
    InlineAsm *UntagAsm = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false),
        StringRef("andq $2, $0"),
        StringRef("=r,0,r,~{flags}"),
        /*hasSideEffects=*/true,
        /*isAlignStack*/false,
        InlineAsm::AD_ATT,
        /*canThrow*/false);
    Value *PtrUntagged = builder.CreateCall(UntagAsm,
        {PtrAsInt, ConstantInt::get(Int64Ty, 0x00FFFFFFFFFFFFFFULL)},
        "subsan.range.ptr.untagged");
    Value *ShiftedClean = builder.CreateLShr(PtrUntagged,
        ConstantInt::get(Int64Ty, 4));
    Value *ByteOffClean = builder.CreateShl(ShiftedClean,
        ConstantInt::get(Int64Ty, 3));
    Value *MetaAddrI = builder.CreateAdd(ShadowBase, ByteOffClean,
        "subsan.range.meta_addr.i");
    MetaPtr = builder.CreateIntToPtr(MetaAddrI, Int64PtrTy,
        "subsan.range.meta_ptr");
  } else {
    Value *ShadowI64Base = builder.CreateIntToPtr(ShadowBase, Int64PtrTy);
    MetaPtr = builder.CreateInBoundsGEP(Int64Ty, ShadowI64Base, GranuleIdx,
                                              "subsan.range.meta_ptr");
  }
  Value *Meta = builder.CreateAlignedLoad(Int64Ty, MetaPtr, Align(8),
                                           "subsan.range.meta");
  if (ClSubsanInlineCheckBarrier) {
    InlineAsm *MetaBarrier = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty}, false),
        /*asm=*/"", /*constraints=*/"=r,0",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        InlineAsm::AD_ATT, /*canThrow=*/false);
    Meta = builder.CreateCall(MetaBarrier, {Meta},
                              "subsan.range.meta.barrier");
  }
  Value *EndOfObj = Meta;  // i64 meta value

  Value *EndAsInt = builder.CreatePtrToInt(end, IntPtrTy);
  TypeSize size = DL.getTypeStoreSize(ptrType);
  Value *Ub = builder.CreateAdd(
      EndAsInt, ConstantInt::get(Int64Ty, size > 0 ? size - 1 : 0),
      "subsan.range.ub");

  // diff = meta - ub; masked = diff & 0xFFFF_<<48; oob = masked != 0
  Value *Diff = builder.CreateSub(EndOfObj, Ub, "subsan.range.diff");
  Value *Masked = builder.CreateAnd(
      Diff, ConstantInt::get(Int64Ty, subsanFastPathMask),
      "subsan.range.masked");
  Value *cmp = builder.CreateICmp(
      CmpInst::Predicate::ICMP_NE, Masked, ConstantInt::get(Int64Ty, 0),
      "subsan.range.oob");

  if(Instruction *RangeChkCmp = dyn_cast<Instruction>(cmp)){
    RangeChkCmp->setMetadata(F->getParent()->getMDKindID("scev_range_chk"), llvm::MDNode::get(F->getContext(), std::nullopt));
  }

  // control flow split location (before the target load/store)
  Instruction *split = &*std::next(cast<Instruction>(cmp)->getIterator());
  
  LLVMContext* CC = &(F->getContext());

  // if-then branch goes to error handling
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false, MDBuilder(*CC).createBranchWeights(1, 10000000), &DT, &LI, nullptr);
  builder.SetInsertPoint(endOfThen);

  if (!ClSubsanCcmpFastPath) {
    Tag = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, tag_shift),
                              "subsan.range.tag");
  }

  // Slow check if tag is zero, then we must be dealing with uninstrumented pointers, ignore
  Value *SlowCheckNonZeroTag = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, Tag, ConstantInt::get(Int64Ty, 0));

  BasicBlock *Head = endOfThen->getParent(); // splitbefore BB
  BasicBlock *Tail = BasicBlock::Create(*CC, "", F, Head->getNextNode());
  new UnreachableInst(*CC, Tail);
  // BasicBlock *Tail = Head->splitBasicBlock(endOfThen->getIterator());


  // FIXME: update DTU for this block?
  if (Loop *L = LI.getLoopFor(Head)) {
    L->addBasicBlockToLoop(Tail, LI);
  }

  Instruction *HeadOldTerm = Head->getTerminator();
  // Instruction *CheckTerm = I.getParent()->getTerminator(); // Then
  BranchInst *HeadNewTerm = BranchInst::Create(/*ifTrue*/ I.getParent(), /*ifFalse*/ Tail, SlowCheckNonZeroTag);
  HeadNewTerm->setMetadata(LLVMContext::MD_prof, MDBuilder(*CC).createBranchWeights(1, 10000000));
  ReplaceInstWithInst(HeadOldTerm, HeadNewTerm);
  // CheckTerm == EndOfThen

  // on Tail: Insert error
  Instruction *NewFailureBlock = Tail->getTerminator();
  builder.SetInsertPoint(NewFailureBlock);

#if 0
  // this call is optional if more informative error reports are desired
  builder.CreateCall(swiftsan_report_fn, {Target, TargetFull, EndOfObj}, "swiftsan_report");
#endif

  if(isX86) {
    // x86 software interrupt
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("int3"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
  else {
    // arm software breakpoint
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("brk #0x0"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
}

std::tuple<Value *, Value *> SafeStack::InsertCheckMeta(Instruction &I, Value &addr, bool write, Type* ptrType, Value *EndOfObj, Value *Tag, bool slowZero) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);

  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  IntegerType *Int64Ty = Type::getInt64Ty(M->getContext());

  builder.SetInsertPoint(&I);

  Value *Target = &addr;

  // ub = ptr + (size - 1)
  Value *PtrAsInt = builder.CreatePtrToInt(Target, IntPtrTy);
  Value *Ub;
  if (ptrType != nullptr) {
    TypeSize size = DL.getTypeStoreSize(ptrType);
    Value *AccessSize = ConstantInt::get(Int64Ty, size > 0 ? size - 1 : 0);
    Ub = builder.CreateAdd(PtrAsInt, AccessSize, "subsan.meta.ub");
  } else {
    Ub = PtrAsInt;
  }

  Value *MetaI64 = EndOfObj;
  if (EndOfObj->getType()->isPointerTy())
    MetaI64 = builder.CreatePtrToInt(EndOfObj, Int64Ty, "subsan.meta.cast");

  // diff = meta - ub; masked = diff & 0xFFFF_<<48; oob = masked != 0
  Value *Diff = builder.CreateSub(MetaI64, Ub, "subsan.meta.diff");
  Value *Masked = builder.CreateAnd(
      Diff, ConstantInt::get(Int64Ty, subsanFastPathMask),
      "subsan.meta.masked");
  Value *cmp = builder.CreateICmp(
      CmpInst::Predicate::ICMP_NE, Masked, ConstantInt::get(Int64Ty, 0),
      "subsan.meta.oob");

  if(Instruction *MetaChkCmp = dyn_cast<Instruction>(cmp)){
    if(slowZero)
      MetaChkCmp->setMetadata(F->getParent()->getMDKindID("meta_chk_slowzero"), llvm::MDNode::get(F->getContext(), std::nullopt));
  }

  // control flow split location (before the target load/store)
  Instruction *split = &*std::next(cast<Instruction>(cmp)->getIterator());

  LLVMContext* CC = &(F->getContext());

  // if-then branch goes to error handling
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false, MDBuilder(*CC).createBranchWeights(1, 10000000), &DT, &LI, nullptr);
  builder.SetInsertPoint(endOfThen);

  // if the metadata was recovered from a stack object size, it is known (and Tag is nullptr)
  // if it is reused from another load, it could be uninstrumented still and we have to slowcheck
  if(Tag || slowZero){

    if(slowZero){
      // a special slow check case for reusing metadata where the tag is unknown and should be calculated from the ptr
      // the pointer is known from the cached metadata, but the tag is not kept around
      uint64_t tag_shift = isImplicitTagging ? 41 : subsanTagShift;

      Value *PtrAsIntSlow = builder.CreatePtrToInt(Target, IntPtrTy);
      Tag = builder.CreateLShr(PtrAsIntSlow,
                               ConstantInt::get(Int64Ty, tag_shift));
    }

    // Slow check if tag is zero, then we must be dealing with uninstrumented pointers, ignore
    Value *SlowCheckNonZeroTag = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, Tag, ConstantInt::get(Int64Ty, 0));

    BasicBlock *Head = endOfThen->getParent(); // splitbefore BB
    BasicBlock *Tail = BasicBlock::Create(*CC, "", F, Head->getNextNode());
    new UnreachableInst(*CC, Tail);
    
    if (Loop *L = LI.getLoopFor(Head)) {
      L->addBasicBlockToLoop(Tail, LI);
    }

    Instruction *HeadOldTerm = Head->getTerminator();
    // Instruction *CheckTerm = I.getParent()->getTerminator(); // Then
    BranchInst *HeadNewTerm = BranchInst::Create(/*ifTrue*/ I.getParent(), /*ifFalse*/ Tail, SlowCheckNonZeroTag);
    HeadNewTerm->setMetadata(LLVMContext::MD_prof, MDBuilder(*CC).createBranchWeights(1, 10000000));
    ReplaceInstWithInst(HeadOldTerm, HeadNewTerm);
    // CheckTerm == EndOfThen

    // on Tail: Insert error
    Instruction *NewFailureBlock = Tail->getTerminator();
    builder.SetInsertPoint(NewFailureBlock);
  }

#if 0
  // this call is optional if more informative error reports are desired
  builder.CreateCall(swiftsan_report_fn, {Target, TargetFull, EndOfObj}, "swiftsan_report");
#endif

  if(isX86) {
    // x86 software interrupt
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("int3"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
  else {
    // arm software breakpoint
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("brk #0x0"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
  return {EndOfObj, Tag};
}

std::tuple<Value *, Value *> SafeStack::InsertCheck(Instruction &I, Value &addr, bool write, Type* ptrType) {
  Function *F = I.getParent()->getParent();
  Module *M = F->getParent();
  LLVMContext &C = F->getContext();
  IRBuilder<> builder(C);

  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  IntegerType *Int64Ty = Type::getInt64Ty(M->getContext());
  Type *Int64PtrTy = PointerType::get(Int64Ty, 0);

  builder.SetInsertPoint(&I);

  Value *Target = &addr;

  uint64_t tag_shift = isImplicitTagging ? 41 : subsanTagShift;
  Value *PtrAsInt = builder.CreatePtrToInt(Target, IntPtrTy);
  Value *Tag = nullptr;
  if (ClSubsanCcmpFastPath) {
    Tag = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, tag_shift),
                              "subsan.tag");
  }

  Value *UbEarly = nullptr;
  if (ClSubsanUbBarrier && ptrType != nullptr) {
    TypeSize size = DL.getTypeStoreSize(ptrType);
    if (size > 1) {
      Value *AccessSize = ConstantInt::get(Int64Ty, size - 1);
      UbEarly = builder.CreateAdd(PtrAsInt, AccessSize, "subsan.ub.early");
      InlineAsm *Barrier = InlineAsm::get(
          FunctionType::get(Int64Ty, {Int64Ty}, false),
          /*asm=*/"",
          /*constraints=*/"=r,0",
          /*hasSideEffects=*/true,
          /*isAlignStack=*/false,
          InlineAsm::AD_ATT,
          /*canThrow=*/false);
      UbEarly = builder.CreateCall(Barrier, {UbEarly}, "subsan.ub.early.barrier");
    }
  }

  Value *GranuleIdx;
  if (isX86 && ClSubsanLamBmi2Bextr) {
    unsigned MaskLen = llvm::countr_one(subsanGranuleIdxMask);
    uint64_t Ctrl = ((uint64_t)MaskLen << 8) | 4ULL;
    InlineAsm *BextrAsm = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false),
        StringRef("bextrq $2, $1, $0"),
        StringRef("=r,r,r,~{flags}"),
        /*hasSideEffects=*/false,
        /*isAlignStack*/false,
        InlineAsm::AD_ATT,
        /*canThrow*/false);
    GranuleIdx = builder.CreateCall(BextrAsm,
        {PtrAsInt, ConstantInt::get(Int64Ty, Ctrl)},
        "subsan.granule_idx");
  } else {
    Value *Shifted = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, 4));
    GranuleIdx = builder.CreateAnd(
        Shifted, ConstantInt::get(Int64Ty, subsanGranuleIdxMask),
        "subsan.granule_idx");
  }

  Value *ShadowBase;
  if (ClSubsanHoistShadowBase) {
    ShadowBase = getOrCreateFnShadowBase();
  } else if (ClSubsanFixedShadowBase) {
    ShadowBase = ConstantInt::get(Int64Ty, ClSubsanShadowOffset);
  } else {
    Constant *ShadowBaseC =
        M->getOrInsertGlobal("__hwasan_shadow_memory_dynamic_address", Int64Ty);
    GlobalVariable *ShadowBaseGV = dyn_cast<GlobalVariable>(ShadowBaseC);
    if (ShadowBaseGV && !ShadowBaseGV->hasInitializer())
      ShadowBaseGV->setLinkage(GlobalValue::ExternalLinkage);
    ShadowBase = builder.CreateAlignedLoad(
        Int64Ty, ShadowBaseGV, Align(8), "subsan.shadow_base");
    cast<LoadInst>(ShadowBase)->setMetadata(
        LLVMContext::MD_invariant_load, MDNode::get(C, std::nullopt));
  }

  Value *MetaPtr;
  if (isX86 && ClSubsanLamBmi2Bextr) {
    Value *ShadowI64Base = builder.CreateIntToPtr(ShadowBase, Int64PtrTy);
    MetaPtr = builder.CreateInBoundsGEP(Int64Ty, ShadowI64Base, GranuleIdx,
                                          "subsan.meta_ptr");
  } else if (isX86) {
    InlineAsm *UntagAsm = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty, Int64Ty}, false),
        StringRef("andq $2, $0"),
        StringRef("=r,0,r,~{flags}"),
        /*hasSideEffects=*/true,
        /*isAlignStack*/false,
        InlineAsm::AD_ATT,
        /*canThrow*/false);
    Value *PtrUntagged = builder.CreateCall(UntagAsm,
        {PtrAsInt, ConstantInt::get(Int64Ty, 0x00FFFFFFFFFFFFFFULL)},
        "subsan.ptr.untagged");
    Value *ShiftedClean = builder.CreateLShr(PtrUntagged,
        ConstantInt::get(Int64Ty, 4));
    Value *ByteOffClean = builder.CreateShl(ShiftedClean,
        ConstantInt::get(Int64Ty, 3));
    Value *MetaAddrI = builder.CreateAdd(ShadowBase, ByteOffClean,
        "subsan.meta_addr.i");
    MetaPtr = builder.CreateIntToPtr(MetaAddrI, Int64PtrTy,
        "subsan.meta_ptr");
  } else {
    Value *ShadowI64Base = builder.CreateIntToPtr(ShadowBase, Int64PtrTy);
    MetaPtr = builder.CreateInBoundsGEP(Int64Ty, ShadowI64Base, GranuleIdx,
                                              "subsan.meta_ptr");
  }
  Value *Meta = builder.CreateAlignedLoad(Int64Ty, MetaPtr, Align(8),
                                           "subsan.meta");

  if (ClSubsanInlineCheckBarrier) {
    InlineAsm *MetaBarrier = InlineAsm::get(
        FunctionType::get(Int64Ty, {Int64Ty}, false),
        /*asm=*/"", /*constraints=*/"=r,0",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        InlineAsm::AD_ATT, /*canThrow=*/false);
    Meta = builder.CreateCall(MetaBarrier, {Meta}, "subsan.meta.barrier");
  }

  if (ClSubsanPrefetchShadow) {
    Type *I32Ty = Type::getInt32Ty(C);
    Value *NextMetaPtr = builder.CreateInBoundsGEP(
        Int64Ty, MetaPtr, ConstantInt::get(Int64Ty, 1), "subsan.next_meta_ptr");
    Function *PrefetchFn = Intrinsic::getDeclaration(
        M, Intrinsic::prefetch, NextMetaPtr->getType());
    CallInst *PrefetchCall = builder.CreateCall(
        PrefetchFn, {NextMetaPtr,
                     ConstantInt::get(I32Ty, 0),  // rw=0 (read)
                     ConstantInt::get(I32Ty, 3),  // locality=3 (high, L1)
                     ConstantInt::get(I32Ty, 1)}); // cache=1 (data)
    PrefetchCall->setMetadata(F->getParent()->getMDKindID("swiftsan"),
                               MDNode::get(C, std::nullopt));
  }

  Value *EndOfObj = Meta;

  // ub = ptr + (size - 1)
  Value *Ub;
  if (UbEarly != nullptr) {
    Ub = UbEarly;
  } else if (ptrType != nullptr) {
    TypeSize size = DL.getTypeStoreSize(ptrType);
    Value *AccessSize = ConstantInt::get(Int64Ty, size > 0 ? size - 1 : 0);
    Ub = builder.CreateAdd(PtrAsInt, AccessSize, "subsan.ub");
  } else {
    Ub = PtrAsInt;
  }

  // diff = meta - ub; masked = diff & 0xFFFF_0000_0000_0000; oob = masked != 0
  Value *Diff = builder.CreateSub(EndOfObj, Ub, "subsan.diff");
  Value *Masked = builder.CreateAnd(
      Diff, ConstantInt::get(Int64Ty, subsanFastPathMask),
      "subsan.masked");
  Value *cmp = builder.CreateICmp(
      CmpInst::Predicate::ICMP_NE, Masked, ConstantInt::get(Int64Ty, 0),
      "subsan.oob");

  // control flow split location (before the target load/store)
  Instruction *split = &*std::next(cast<Instruction>(cmp)->getIterator());

  LLVMContext* CC = &(F->getContext());

  // if-then branch goes to error handling
  Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false, MDBuilder(*CC).createBranchWeights(1, 10000000), &DT, &LI, nullptr);
  builder.SetInsertPoint(endOfThen);

  if (!ClSubsanCcmpFastPath) {
    Tag = builder.CreateLShr(PtrAsInt, ConstantInt::get(Int64Ty, tag_shift),
                              "subsan.tag");
  }

  // Slow check if tag is zero, then we must be dealing with uninstrumented pointers, ignore
  Value *SlowCheckNonZeroTag = builder.CreateICmp(CmpInst::Predicate::ICMP_EQ, Tag, ConstantInt::get(Int64Ty, 0));

  BasicBlock *Head = endOfThen->getParent(); // splitbefore BB

  // BasicBlock *Tail = Head->splitBasicBlock(endOfThen->getIterator());

  BasicBlock *Tail = BasicBlock::Create(*CC, "", F, Head->getNextNode());
  new UnreachableInst(*CC, Tail);
  
  if (Loop *L = LI.getLoopFor(Head)) {
    L->addBasicBlockToLoop(Tail, LI);
  }

  Instruction *HeadOldTerm = Head->getTerminator();
  // Instruction *CheckTerm = I.getParent()->getTerminator(); // Then
  BranchInst *HeadNewTerm = BranchInst::Create(/*ifTrue*/ I.getParent(), /*ifFalse*/ Tail, SlowCheckNonZeroTag);
  HeadNewTerm->setMetadata(LLVMContext::MD_prof, MDBuilder(*CC).createBranchWeights(1, 10000000));
  ReplaceInstWithInst(HeadOldTerm, HeadNewTerm);
  // CheckTerm == EndOfThen

  // on Tail: Insert error
  Instruction *NewFailureBlock = Tail->getTerminator();
  builder.SetInsertPoint(NewFailureBlock);

#if 0
  // this call is optional if more informative error reports are desired
  builder.CreateCall(swiftsan_report_fn, {Target, TargetFull, EndOfObj}, "swiftsan_report");
#endif

  if(isX86) {
    // x86 software interrupt
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("int3"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
  else {
    // arm software breakpoint
    InlineAsm *IA = InlineAsm::get(
                    FunctionType::get(llvm::Type::getVoidTy(C), {}, false),
                    StringRef("brk #0x0"),
                    StringRef(""),
                    /*hasSideEffects=*/ true,
                    /*isAlignStack*/ false,
                    InlineAsm::AD_ATT,
                    /*canThrow*/ false);
    builder.CreateCall(IA, {});
  }
  return {EndOfObj, Tag};
}

// isSafeAccess returns true if Addr is always inbounds with respect to its
// base object. For example, it is a field access or an array access with
// constant inbounds index.
bool SafeStack::isSafeAccess(ObjectSizeOffsetVisitor &ObjSizeVis, Value *Addr,
                  uint64_t TypeSize) {
  SizeOffsetType SizeOffset = ObjSizeVis.compute(Addr);
  if (!ObjSizeVis.bothKnown(SizeOffset))
    return false;
  uint64_t Size = SizeOffset.first.getZExtValue();
  int64_t Offset = SizeOffset.second.getSExtValue();
  // Three checks are required to ensure safety:
  // . Offset >= 0  (since the offset is given from the base ptr)
  // . Size >= Offset  (unsigned)
  // . Size - Offset >= NeededSize  (unsigned)
  return Offset >= 0 && Size >= uint64_t(Offset) &&
          Size - uint64_t(Offset) >= TypeSize / 8;
}

//ASAN--: Removing Unsatisfiable Checks
bool SafeStack::isSafeAccessBoost(ObjectSizeOffsetVisitor &ObjSizeVis, Instruction *IndexInst, Value *Addr, Function *F) const {
  auto DT = DominatorTree(*F);
  if (GetElementPtrInst *Gep_Inst = dyn_cast<GetElementPtrInst>(Addr)) {
    for (auto& Index : make_range(Gep_Inst->idx_begin(), Gep_Inst->idx_end())) {
      for (User *U : Index->users()) {   
        if (CmpInst *i_cmp = dyn_cast<CmpInst>(U)) {
          if (DT.dominates(i_cmp, IndexInst)) {
            if (Index == i_cmp->getOperand(0) && isa<ConstantData>(i_cmp->getOperand(1))) {
              auto IndexSize = i_cmp->getOperand(1);
              auto ConstantSize = dyn_cast<ConstantInt>(IndexSize);
              int64_t MaxOffset = ConstantSize->getSExtValue();
              auto type = Gep_Inst->getPointerOperandType();

              if (cast<PointerType>(type)) {
                auto pttpee = Gep_Inst->getSourceElementType();
                if (isa<ArrayType>(pttpee)) {
                  auto ObjSize = pttpee->getArrayNumElements();
                  return static_cast<int64_t>(ObjSize) >= MaxOffset;
                }
              }
              if (isa<ArrayType>(type)) {
                auto ObjSize = type->getArrayNumElements();
                return static_cast<int64_t>(ObjSize) >= MaxOffset;
              }
            }

            if (Index == i_cmp->getOperand(1) && isa<ConstantData>(i_cmp->getOperand(0))) {
              auto IndexSize = i_cmp->getOperand(0);
              auto ConstantSize = dyn_cast<ConstantInt>(IndexSize);
              int64_t MaxOffset = ConstantSize->getSExtValue();
              auto type = Gep_Inst->getPointerOperandType();
              if (cast<PointerType>(type)) {
                auto pttpee = Gep_Inst->getSourceElementType();
                if (isa<ArrayType>(pttpee)) {
                  auto ObjSize = pttpee->getArrayNumElements();
                  return static_cast<int64_t>(ObjSize) >= MaxOffset;
                }
              }
              if (isa<ArrayType>(type)) {
                auto ObjSize = type->getArrayNumElements();
                return static_cast<int64_t>(ObjSize) >= MaxOffset;
              }
            }
          }
        }
      } 
    }
  }
  return false;
}

void SafeStack::OptStaticallySafeAccesses(ObjectSizeOffsetVisitor &ObjSizeVis, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument) {
  SmallVector<InterestingMemoryOperand, 16> TempToInstrument(OperandsToInstrument);
  TempToInstrument.clear();
  bool toKeep;
  for (InterestingMemoryOperand &O : OperandsToInstrument) {
    Value *Addr = O.getPtr();
    Instruction *Insn = O.getInsn();
    toKeep = true;

    // If global variable
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Addr))) {
      if (isSafeAccess(ObjSizeVis, Addr, O.TypeSize)) {
        toKeep = false;
      }
      // if we can determine that this load/store was on a skipped global, no need to check
      else if(!shouldInstrumentGlobal(*GV)){
        toKeep = false;
      }
      //ASAN--: Removing Unsatisfiable Checks
      if (isSafeAccessBoost(ObjSizeVis, Insn, Addr, Insn->getFunction())) {
        toKeep = false;
      }
    }

    // If stack variable
    if (isa<AllocaInst>(getUnderlyingObject(Addr))) {
      // All remaining AllocaInst are safe, because otherwise they are already moved to UnsafeStack
      toKeep = false;
    }

    if (toKeep) {
      TempToInstrument.push_back(O);
    }
  }

  // Delete the instructions to avoid
  OperandsToInstrument.clear();
  for (auto item : TempToInstrument) {
    OperandsToInstrument.push_back(item);
  }
}

void SafeStack::OptStaticallySafeAllocas(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument) {
  // this should be redundant, but lets double check
  // all load stores in SafeMemOps are safe (they are users of Allocas that are on the safe stack)
  SmallVector<InterestingMemoryOperand, 16> TempToInstrument(OperandsToInstrument);
  TempToInstrument.clear();
  bool toKeep;

  for (InterestingMemoryOperand &O : OperandsToInstrument) {
    toKeep = true;
    Instruction *targetInsn = O.getInsn();
    for (Instruction *SafeI : SafeMemOps){
        if(SafeI == targetInsn){
          toKeep = false;
        }
    }

    if (toKeep) {
      TempToInstrument.push_back(O);
    }
  }

  // Delete the instructions to avoid
  OperandsToInstrument.clear();
  for (auto item : TempToInstrument) {
    OperandsToInstrument.push_back(item);
  }
}

void SafeStack::OptStaticallySafeUnsafeStack(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument) {

  // Two patterns to look for:
  // #1: load/store directly on the start of the alloca: always safe
  // %680 = getelementptr i8, ptr %9, i32 -2048, !dbg !94300, !unsafe.stack.og_size !35499
  // store i8 0, ptr %680, align 16, !dbg !94300, !tbaa !19129

  // #2: load/store on an offset on the start of the alloca: check in bounds
  // %345 = getelementptr i8, ptr %7, i32 -384, !dbg !16323, !unsafe.stack.og_size !15909
  // %346 = getelementptr inbounds i8, ptr %345, i64 16, !dbg !16323
  // load/store %436

  // note that with pointer tagging, there is an OR after the stack allocation for the tag
  std::set<Instruction *> optimized; 

  for (InterestingMemoryOperand &O : OperandsToInstrument) {

    // take the load/store ptr:
    Value *Addr = O.getPtr()->stripPointerCasts();

    // case 1: load/store directly on unsafe stack obj (includes pointer tag OR-operation)
    if(Constant *Sz = getUnsafeStackObjOgSize(Addr)){
      // this is trivially safe (assuming the access size is valid)
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Sz)){
        APInt SzInt = CI->getValue();
        // access size
        APInt AccSize(SzInt.getBitWidth(), DL.getTypeStoreSize(O.OpType), false); // unsafe obj size is never negative
        if(AccSize.ult(SzInt)){
          optimized.insert(O.getInsn());
          continue;
        }
      }
    }

    // case 2: it is a constant-offset gep of which the ptr is the stack obj (avoid multi-chain GEP offsets pointer arith.)
    if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr)){
      // we can only prove safe if the offset is constant
      if(!GEP->hasAllConstantIndices()) continue;
      APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0, true);
      if(!GEP->accumulateConstantOffset(DL, GEPOffset)) continue;

      // this would be quite strange. it would immediately go OOB underflow on the unsafe stack obj
      if(GEPOffset.isNegative()) continue;

      // the next pointer operand needs to be the unsafe stack object (exclude repeated GEPs for now)
      Value *VPtr = GEP->getPointerOperand()->stripPointerCasts();
      Constant *Sz = getUnsafeStackObjOgSize(VPtr);
      if(Sz == nullptr) continue;

      // Compare Sz to GEPOffset: it should be ConstantInt
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Sz)){
        APInt SzInt = CI->getValue();
        if(GEPOffset.slt(SzInt)){ // e.g. idx 63 is max for sz 64
          optimized.insert(O.getInsn());
          continue;
        }
      }
    }
  }

	SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}

}

void rankRemovableInsts(std::map<Instruction *, std::set<std::pair<Instruction *, Instruction *>>> &potentialRemoveInsts, std::list<std::pair<int, Instruction *>> &rankPotentialRemoveInsts) {
  for(auto instVectorMap = potentialRemoveInsts.begin(); instVectorMap != potentialRemoveInsts.end(); ++instVectorMap) {
    int countInst = 0;
    for(auto instVector = potentialRemoveInsts.begin(); instVector != potentialRemoveInsts.end(); ++instVector) {
      if (instVector == instVectorMap)
        continue;
      for (auto instPair = (*instVector).second.begin(); instPair != (*instVector).second.end(); ++instPair) {
        if ((*instVector).first == instPair->first || (*instVector).first == instPair->second) {
          countInst++;
        }
      } 
    }
    rankPotentialRemoveInsts.push_back(std::pair<int, Instruction *>(countInst, (*instVectorMap).first));
  }
}

void removeInstructionFunc(std::map<Instruction *, std::set<std::pair<Instruction *, Instruction *>>> &potentialRemoveInsts, std::set<Instruction *> &deleted) {
  
  std::list<std::pair<int, Instruction *>> rankPotentialRemoveInsts;

  rankRemovableInsts(potentialRemoveInsts, rankPotentialRemoveInsts);

  rankPotentialRemoveInsts.sort();

  for(auto countInst : rankPotentialRemoveInsts) {
    bool removeInst = true;
    for (auto elem : deleted) {
      removeInst = false;
      for (auto instPair : potentialRemoveInsts[countInst.second]) {
        if (instPair.first != elem && instPair.second != elem) {
          removeInst = true;
          break;
        }
      }
      if (!removeInst)
        break;
    }
    if (!removeInst)
      continue;
    deleted.insert(countInst.second);
    // remove all pairs that contain current key instruction and update the map
    for(auto instVectorMap = potentialRemoveInsts.begin(); instVectorMap != potentialRemoveInsts.end(); ++instVectorMap) {
      for (auto instPair = (*instVectorMap).second.begin(); instPair != (*instVectorMap).second.end();) {
        if (countInst.second == instPair->first || countInst.second == instPair->second) {
          instPair = (*instVectorMap).second.erase(instPair);
        }
        else {
          ++instPair;
        }
      } 
    }
  }
}


void preprocessPotentialRemoveInsts(Function &F, std::pair<const std::pair<llvm::Value *, std::string>, std::set<std::pair<int64_t, llvm::Instruction *>>> &baseAddrOffsetSet, std::map<Instruction *, std::set<std::pair<Instruction *, Instruction *>>> &potentialRemoveInsts) {
  
  auto DT = DominatorTree(F);

  auto PDT = PostDominatorTree();

  PDT.recalculate(F);
  
  // offsetInstA is node A
  for (auto offsetInstA : baseAddrOffsetSet.second) {
    // offsetInstB is node B
    for (auto offsetInstB : baseAddrOffsetSet.second) {
      if (offsetInstA == offsetInstB)
        continue;
      // offsetInstC is node C
      for (auto offsetInstC : baseAddrOffsetSet.second) {
        if (offsetInstA == offsetInstC || offsetInstB == offsetInstC)
          continue;

        // Here we ensure (A dominate B OR A post-dominate B) AND (OFFSET(C) > OFFSET(B) AND OFFSET(B) > OFFSET(A) AND OFFSET(C) - OFFSET(A) < 16)
        if ( (DT.dominates(offsetInstA.second, offsetInstB.second) || PDT.dominates((offsetInstA.second)->getParent(), (offsetInstB.second)->getParent())) 
            && (offsetInstC.first > offsetInstB.first && offsetInstB.first > offsetInstA.first && offsetInstC.first - offsetInstA.first < RZ_SIZE) ) {
          // If above conditions are satisfied, then ASan check on B can be removed.
          if (potentialRemoveInsts.find(offsetInstB.second) == potentialRemoveInsts.end()) {
            potentialRemoveInsts.insert(std::pair<Instruction *, std::set<std::pair<Instruction *, Instruction *>>>(offsetInstB.second, std::set<std::pair<Instruction *, Instruction *>>()));
          }
          // Store the ASan check removable instruction B, and the pair of instructions A and C that ensure the ASan Check to map
          std::pair<Instruction *, Instruction *> InstsPair;
          InstsPair.first = offsetInstA.second;
          InstsPair.second = offsetInstC.second;
          potentialRemoveInsts[offsetInstB.second].insert(InstsPair);
        } 
      }
    }
  }
}

void rmNeighborChks(Function &F,std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi, std::set<Instruction *> &deleted) {

  for (auto baseAddrOffsetSet : baseAddrOffsetMap_multi) {
    // Create a map to store the ASan check removable instruction, and the pair of instruction to ensure the ASan check
    std::map<Instruction *, std::set<std::pair<Instruction *, Instruction *>>> potentialRemoveInsts;
    // Cases for size of set >= 3
    if ((baseAddrOffsetSet.second).size() >=3 ) {
      preprocessPotentialRemoveInsts(F, baseAddrOffsetSet, potentialRemoveInsts);
      removeInstructionFunc(potentialRemoveInsts, deleted); 
    }
  }
}

void singleIndexCaseHandler(std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi, GetElementPtrInst *Gep_Inst, Instruction *Inst) {
  
  Value *baseAddr = Gep_Inst->getPointerOperand();
  // In order to make form unified, we create a string place holder
  std::string offsets_single;
  std::pair<Value *, std::string> key;
  std::pair<int64_t, Instruction *> value;

  if (auto *offsetAddr = dyn_cast<ConstantInt>(Gep_Inst->idx_begin())) {
    key.first = baseAddr;
    key.second = offsets_single;
    if (baseAddrOffsetMap_multi.find(key) == baseAddrOffsetMap_multi.end()) {
      //never appeared in the map, so add a slot
      baseAddrOffsetMap_multi.insert(std::pair<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *> >>(key, std::set<std::pair<int64_t, Instruction *>>()));
    }
    // Convert last offset into int
    int64_t intLastOffset = offsetAddr->getSExtValue();
    value.first = intLastOffset;
    value.second = Inst;
    baseAddrOffsetMap_multi[key].insert(value);
  }
  return;
}

void multiIndexCaseHandler(std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi, GetElementPtrInst *Gep_Inst, Instruction *Inst) {
      
  Value *baseAddr = Gep_Inst->getPointerOperand();
  std::pair<Value *, std::string> key;
  std::pair<int64_t, Instruction *> value;
  
  // String to collect offsets from beg to end - 1
  std::string offsets;
  bool offsetConstantInt = true;
  for (auto& index : make_range(Gep_Inst->idx_begin(), Gep_Inst->idx_end() - 1)) {
    if (auto *offsetAddr_multi = dyn_cast<ConstantInt>(index)) {
      int64_t intOffset = offsetAddr_multi->getSExtValue();
      offsets.push_back(intOffset);
    } else {
      offsetConstantInt = false;
      break;
    }
  }

  if (!offsetConstantInt) {
    return;
  }
  
  // Here we check the value of last offset
  if (auto *offsetAddr_last = dyn_cast<ConstantInt>(Gep_Inst->idx_end() - 1)) {
    key.first = baseAddr;
    key.second = offsets;
    if (baseAddrOffsetMap_multi.find(key) == baseAddrOffsetMap_multi.end()) {
      //never appeared in the map, so add a slot
        baseAddrOffsetMap_multi.insert(std::pair<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>>(key, std::set<std::pair<int64_t, Instruction *>>()));
    }
    // Convert last offset into int
    int64_t intLastOffset = offsetAddr_last->getSExtValue();
    value.first = intLastOffset;
    value.second = Inst;
    baseAddrOffsetMap_multi[key].insert(value);
  }
  return;
}

void SafeStack::baseAddrOffsetMapPreprocessing(SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi) {

  for (auto oper : OperandsToInstrument) {

    Value *addr = oper.getPtr();
    if (!addr)
      continue;

    while (CastInst *Cast_Inst = dyn_cast<CastInst>(addr))
      addr = Cast_Inst->getOperand(0);
    
    // Check if current address is from a gep instruction
    if (GetElementPtrInst *Gep_Inst = dyn_cast<GetElementPtrInst>(addr)) {

      if (Gep_Inst->getNumIndices() == 1) {
        singleIndexCaseHandler(baseAddrOffsetMap_multi, Gep_Inst, oper.getInsn());
        continue;
      }
      multiIndexCaseHandler(baseAddrOffsetMap_multi, Gep_Inst, oper.getInsn());
      continue;
    }
  }
}

void updateBaseAddrOffsetMap(std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> &baseAddrOffsetMap_multi, std::set<Instruction *> &deleted) {
  for (auto baseAddrOffsetSet = baseAddrOffsetMap_multi.begin(); baseAddrOffsetSet != baseAddrOffsetMap_multi.end(); ++baseAddrOffsetSet) {
    for (auto offsetInst = (*baseAddrOffsetSet).second.begin(); offsetInst != (*baseAddrOffsetSet).second.end();) {
      if (deleted.find((*offsetInst).second) != deleted.end()) {
        offsetInst = (*baseAddrOffsetSet).second.erase(offsetInst);
      }
      else {
        ++offsetInst;
      }
    }
  }
}

void SafeStack::sequentialExecuteOptimizationBoost(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument) {

  std::map<std::pair<Value *, std::string>, std::set<std::pair<int64_t, Instruction *>>> baseAddrOffsetMap_multi;

  baseAddrOffsetMapPreprocessing(OperandsToInstrument, baseAddrOffsetMap_multi);

  std::set<Instruction *> deleted;

  // ASAN--: Removing Neighbor Checks
  rmNeighborChks(F, baseAddrOffsetMap_multi, deleted);

  updateBaseAddrOffsetMap(baseAddrOffsetMap_multi, deleted);

  // ASAN--: Merging Neighbor Checks
  // skip. not applicable

  SmallVector<InterestingMemoryOperand, 16> SEOTempToInstrument(OperandsToInstrument);
  OperandsToInstrument.clear();

  for (auto item: SEOTempToInstrument) {
    if (deleted.find(item.getInsn()) == deleted.end())
      OperandsToInstrument.push_back(item);
  }
}

enum addrType SafeStack::loopOptimizationCategorise(Function &F, Loop *L, InterestingMemoryOperand Oper, ScalarEvolution *SE) {

  std::vector<Value *> backs;
  std::vector<Value *> processedAddr;

  if (Value* addr = Oper.getPtr()) {
	  btraceInLoop(addr, backs, L);
	  return checkAddrType(addr, backs, processedAddr, SE, L);
  }
  return UNKNOWN; 
}

void SafeStack::ConservativeCallIntrinsicCollect(Function &F, std::set<Instruction *> &callIntrinsicSet) {

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

bool SafeStack::isPostDominatWrapper(Instruction *InstStart, Instruction *TargetInst, llvm::PostDominatorTree &PDT) {
  
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

bool SafeStack::ConservativeCallIntrinsicCheck(Instruction *InstStart, Instruction *InstEnd, std::set<Instruction *> &callIntrinsicSet, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT) {

  for (auto TargetInst : callIntrinsicSet) {
    // InstStart -> TargetInst -> InstEnd && InstStart !PostDominat TargetInst
    if (isPotentiallyReachable(InstStart, TargetInst) && isPotentiallyReachable(TargetInst, InstEnd) && !isPostDominatWrapper(InstStart, TargetInst, PDT)) {
      return false;
    }
  }
  return true;
}

void SafeStack::sequentialExecuteOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA) {
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
  }
}

void SafeStack::sequentialExecuteOptimizationPostDom(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, AliasAnalysis *AA) {
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
    if (deleted.find(item.getInsn()) == deleted.end())
      OperandsToInstrument.push_back(item);
  }
}

bool isLoopInvariantValue(Value *V, Loop *L, ScalarEvolution *SE){
  // first try regular Loop invariance
  if(Instruction* IV = dyn_cast<Instruction>(V)){
    // first try the more optimistic 'all operands are invariant'
    // which means the operation should be duplicatable with invariance
    // TODO: recursively explore the operands?
    bool invariant = L->hasLoopInvariantOperands(IV);
    if(invariant) return true;
  }

  // if not an instruction, maybe it lives outside the loop?
  bool invariant = L->isLoopInvariant(V);
  if(invariant) return true;

  // still not provably invariant, try if SCEV can prove it
  const SCEV *VExpr = SE->getSCEV(V);
  if(!VExpr){
    // fail
    return false;
  }
  return SE->isLoopInvariant(VExpr, L);
}

void SafeStack::ClassifyLoopAccesses(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, 
SmallVector<InterestingMemoryOperand, 16> &invariants, LoopInfo *LI, ScalarEvolution *SE){

  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {
    // if the load/store lives in a loop
    if (Loop *L = LI->getLoopFor(Oper.getInsn()->getParent())) {

      // try multiple strategies to determine invariance
      bool invariantAPI = isLoopInvariantValue(Oper.getPtr(), L, SE);
      if(invariantAPI){
        invariants.push_back(Oper);
        continue;
      }
      
      // existing API says !invariant, try ASan-- categorization
      if (loopOptimizationCategorise(F, L, Oper, SE) == IBIO) {
        invariants.push_back(Oper);
        continue;
      }

      // variant, we do not track.
    }
  }
}

bool isOperInVector(InterestingMemoryOperand &target, SmallVector<InterestingMemoryOperand, 16> &vecc){
  for(auto Oper : vecc){
    if(Oper.getInsn() == target.getInsn()){
      return true;
    }
  }
  return false;
}

bool SafeStack::LoopPtrReuseMetadata(Function &F, LoopRangeCheck &Check, SmallVector<CheckMergeCands, 16> &CheckMergers, SmallVector<CheckMergeCands, 16> &FuncMetaMergers, std::set<Instruction *> &optimized){

  // here: Check.InsertPt == &(*F.begin()->getFirstInsertionPt());

  IRBuilder<> builder(Check.MemAccess); // initially create the builder on the load/store
  LLVMContext* CC = &(F.getContext());

  auto exitBB = Check.L->getExitBlock();

  if(Check.invariant){
    /* Invariant ptr: re-use the check result
      (alloca) bool check_cache = false;
      if(!check_cache){
        full_check(); // <-- aborts if invalid
        check_cache = true;
      }
      *load;

      dominating loads: check at exit
      (alloca) bool reached = false;
      for(...){
        if(...)
          reached = true;
          *load
      }
      if(reached) full_check();
    */

    Instruction *IPtr = dyn_cast<Instruction>(Check.Ptr);
    Instruction *IAccess = dyn_cast<Instruction>(Check.MemAccess);
    bool checkLoadAtExit = exitBB && isa<LoadInst>(Check.MemAccess) && IPtr && DT.dominates(IPtr, exitBB) && DT.dominates(IAccess, exitBB);
    // only apply this optimization in the case of loads that can be checked in an exit block
    // other cases cause bloat too quickly
    if(checkLoadAtExit){

      Type* AllocaType = Type::getInt1Ty(*CC);
      Value* ValUnset = ConstantInt::get(AllocaType, 0);
      Value* ValSet = ConstantInt::get(AllocaType, 1);
      AllocaInst *CheckCache = new AllocaInst(AllocaType, DL.getAllocaAddrSpace(), "BB_CacheCheck", Check.InsertPt);
      CheckCache->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
    
      // initialize the alloca check cache to fail the first loop iteration (in the preheader)
      Instruction *PreheaderInit = &*Check.L->getLoopPreheader()->getTerminator();
      builder.SetInsertPoint(PreheaderInit);
      Instruction *StoreInit = builder.CreateStore(ValUnset, CheckCache);
      StoreInit->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
    
      // loads that dominate the (single) exit block can be checked after the loop
      // at the load, set the flag to true unconditionally, to signal the loop exit
      // to perform the check, since the access must have been executed
      builder.SetInsertPoint(Check.MemAccess);
      Instruction *StoreSet = builder.CreateStore(ValSet, CheckCache);
      StoreSet->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

      // move the builder to the exit block
      builder.SetInsertPoint(&(*exitBB->getFirstInsertionPt()));
      Value *checkValue = builder.CreateLoad(AllocaType, CheckCache);
      Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_NE, checkValue, ValUnset);
      
      // control flow split location: inside the exit block
      Instruction *split = &*std::next(cast<Instruction>(cmp)->getIterator());

      // if-then - then branch (set) goes to check
      Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false, MDBuilder(*CC).createBranchWeights(75, 25), &DT, &LI, nullptr);

      InsertCheck(*endOfThen, *Check.Ptr, false, Check.ptrType);

      // no need to reset the check cache, preheader will re-initialize if we are an inner loop
      // notify the caller to mark the load/store as checked/optimized

      return true;
    }
    return false;
  }
  else{
    /* variant ptr, basically:
      (alloca) uintptr_t meta = 0;
      ...
      if(cur_ptr > meta){
        meta = full_check()->end_of_obj;
      }
      *load;
    */

    // stack obj
    Type* AllocaType = PointerType::get(Type::getInt64Ty(*CC), 0);
    Value* ValUnset = Constant::getNullValue(AllocaType);
    AllocaInst *MetaAlloc = new AllocaInst(AllocaType, DL.getAllocaAddrSpace(), "BB_ReuseMetadataLoop", Check.InsertPt);
    MetaAlloc->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    // initialize the alloca check result to fail the first loop iteration (in the preheader)
    Instruction *PreheaderInit = &*Check.L->getLoopPreheader()->getTerminator();

    builder.SetInsertPoint(PreheaderInit); // Check.L->getLoopPreheader()->getTerminator()
    Instruction *StoreInit = builder.CreateStore(ValUnset, MetaAlloc);
    StoreInit->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    // at the checked location: load the metadata and compare it against the ptr
    builder.SetInsertPoint(Check.MemAccess);

    // cmp load(MetaAlloc) < cur_ptr
    // if true: MetaAlloc = full_check()->EndOfObj
    Value *metaValue = builder.CreateLoad(AllocaType, MetaAlloc);

    Value* CurPtr = Check.Ptr;
    TypeSize size = DL.getTypeStoreSize(Check.ptrType);
    IntegerType *IntPtrTy = DL.getIntPtrType(F.getParent()->getContext());
    if(size > 1) {
      Value *AccessSize = ConstantInt::get(F.getContext(), APInt(IntPtrTy->getBitWidth(), size-1));
      // Offset the to-be-checked address by the access size
      std::vector<Value *> indizes = {AccessSize};
      CurPtr = builder.CreateInBoundsGEP(builder.getInt8Ty(), Check.Ptr, indizes);
    }

    // compare target >= end_addr (unsigned greater or equal)
    Value *cmp = builder.CreateICmp(CmpInst::Predicate::ICMP_UGE, CurPtr, metaValue);

    // control flow split location (before the target load/store)
    Instruction *split = &*std::next(cast<Instruction>(cmp)->getIterator());

    // if-then branch goes to error handling
    Instruction *endOfThen = SplitBlockAndInsertIfThen(cmp, split, /*unreachable*/false, MDBuilder(*CC).createBranchWeights(1, 10000000), &DT, &LI, nullptr);

    // endOfThen means check failed, we have to store the true metadata
    std::tuple<Value*, Value*> EndNTag;
    // pass 'CurPtr' which is already offset, set ptrType = nullptr to notify
    EndNTag = InsertCheck(*endOfThen, *CurPtr, true, nullptr);
    
    // if the true check does not fail, we need to store the metadata for the next iterations
    builder.SetInsertPoint(endOfThen);
    Instruction *CacheMetaStore = builder.CreateStore(std::get<0>(EndNTag), MetaAlloc);
    CacheMetaStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

    // no need to reset the metadata, preheader will re-initialize if we are an inner loop
    // notify the caller to mark the load/store as checked/optimized

    return true;
  }
  return false;
}


bool SafeStack::PtrContainsCallUse(Value *Ptr, Loop *L){

  // if we can confirm Ptr is stack/global memory, then potential dealloc calls are not relevant
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(Ptr))) {
    return false;
  }
  if (isa<AllocaInst>(getUnderlyingObject(Ptr))) {
    return false;
  }
  // unsafestack is also stack
  Value *base = nullptr;
  Constant *og_size = nullptr;
  SmallPtrSet<Value *, 4> Visited;
  AccumulateToUnsafeStackAlloca(Ptr, Visited, &base, &og_size);
  if(base) {
    return false;
  }

  // #1 super conservative: if the loop contains ANY call, do not optimize
  // #2 less conservative: look through all the calls in the loop, take their arguments, and see if they alias the pointer

#if 1 // less conservative
  for (auto *BB : L->getBlocks()) {
      for (auto &I : *BB) {
        if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          if(CB->isDebugOrPseudoInst() || CB->isLifetimeStartOrEnd()){
            continue;
          }
#if 0
          // super conservative: if any call in the loop, no opt
          return true;
#endif

          for (Value *arg : CB->args()){
            AliasResult a = AA->alias(Ptr, arg);
            if(a == AliasResult::MayAlias || a == AliasResult::MustAlias ){
              // less conservative: the target ptr is used in a call (as an alias)
              return true;
            }
          }
        }
      }
  }
#endif

  // the underlying allocation of ptr
  if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr->stripPointerCasts())){
    Value *GEPBase = GEP->getPointerOperand();
    if(PtrContainsCallUse(GEPBase, L)){
      return true;
    }

    // can the result of the GEP be freed? maybe for an array of pointers?
    // return false;
  }

  for (User *U : Ptr->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (!L->contains(CI)) {
        continue;
      }
      // ptr is used by a call in loop, possibly freed
      return true;
    }
  }
  return false;
}

void SafeStack::LoopOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<InterestingMemoryOperand, 16> &invariants, SmallVector<CheckMergeCands, 16> &CheckMergers, SmallVector<CheckMergeCands, 16> &FuncMetaMergers, LoopInfo *LI, ScalarEvolution *SE) { 

  SmallVector<LoopRangeCheck, 32> checks;

  std::set<Instruction *> optimized; 
  for (auto Oper : OperandsToInstrument) {
    // if the load/store lives in a loop
    if (Loop *L = LI->getLoopFor(Oper.getInsn()->getParent())) {

      Value *Addr = Oper.getPtr();

      // if the pointer may be freed in the loop, we cannot optimize
      if(PtrContainsCallUse(Addr, L)){
        continue;
      }

      bool conditional = !isGuaranteedToExecuteForEveryIteration(Oper.getInsn(), L);
      auto ExitBB = L->getExitBlock();

      Instruction *InsertPt = nullptr;
      // first attempt to hoist load operations to the exit block
      // otherwise, fall back to preheader
      // store operations we always have to hoist to the preheader
      // because post-loop (exit block) checks would allow potential metadata corruption
      if (LoadInst *LoadI = dyn_cast<LoadInst>(Oper.getInsn())) {
        // if the loop has an exit block which the loaded pointer value dominates
        // such that the loaded pointer value is known at the exit of the loop
        // this also includes non-conditionally executing the load itself
        // TODO: multiple exit blocks
        Instruction *IAddr = dyn_cast<Instruction>(Addr);
        if(IAddr){
          if(ExitBB && DT.dominates(LoadI, ExitBB) && DT.dominates(IAddr, ExitBB)){
            // guaranteed to execute based on exit domination
            InsertPt = &(*ExitBB->getFirstInsertionPt());
            conditional = false;
          }
        }
        // otherwise, fall back to the preheader
      }

      // if not exit block, default is preheader
      if(!InsertPt){

        // we need a preheader. loopSimplify pass should guarantee one
        if(!L->getLoopPreheader()){
          continue;
        }

        if(conditional){

          // pointers here cannot undergo hoisting
          // because even though the ptr does not change in the loop,
          // it could be invalid/garbage and not touched inside the loop (bc conditions)
          // they can however use check result caching or metadata caching
          // these modifications insert allocas and extra stores, so they can cause bloat

          // Additionally, if we have metadata sharing (non-loop-based), we may find that two loads
          // inside a loop actually access the same base object with pos offset, and they can reuse metadata
          // so in that case the second load can re-use the cached metadata.
          // right now this would result in two allocas storing the same metadata

          Instruction *flagInsertPt = &(*F.begin()->getFirstInsertionPt());

          bool inv = isOperInVector(Oper, invariants);
          if(inv){
            // no need for the rest of the SCEV/path analysis
            continue;
          }
          else{
            const SCEV *PtrExpr = SE->getSCEV(Addr);
            if (auto *AR = dyn_cast<SCEVAddRecExpr>(PtrExpr)) {
              const SCEV *Step = AR->getStepRecurrence(*SE);
              if (SE->isKnownPositive(Step)) {
                // mark as to-be-checked per usual, but with metadata caching
                checks.emplace_back(flagInsertPt, Oper.OpType, nullptr, nullptr, Oper.getInsn(), L, inv, Addr, true);
                // no need for the rest of the SCEV/path analysis
                continue;
              }
            }
          }
          // cannot optimize this: e.g., no AddRec, not known positive, not invariant
          continue;
        }
        else{
          // unconditional (non-exit-block): insertion point is the end of the preheader
          InsertPt = L->getLoopPreheader()->getTerminator();
        }

      } // end of !exit (preheader)

      const SCEV *ScStart = nullptr;
      const SCEV *ScEnd = nullptr;
      bool invariant = false;

      if(isOperInVector(Oper, invariants)){

        // key exists: ptr invariant -> no change in the loop
        // ScStart = ScEnd = PtrExpr;
        invariant = true;

        Instruction *IAddr = dyn_cast<Instruction>(Addr);
        if(!IAddr) continue;

        if(!DT.dominates(IAddr, InsertPt)){
          // this suggests LICM pass did not move this out
          continue;
        }
      }
      else{

        // variant ptr -> get AddRecExpr
        const SCEV *PtrExpr = SE->getSCEV(Addr);
        if(!PtrExpr){
            continue;
        }

        if (auto *AR = dyn_cast<SCEVAddRecExpr>(PtrExpr)) {
          // entry value of the loop
          ScStart = AR->getStart();

          // the SCEV in the scope of the parent loop is the exit value of the current loop
          ScEnd = SE->getSCEVAtScope(AR, L->getParentLoop());

          const SCEV *Step = AR->getStepRecurrence(*SE);
          // for expressions with negative step, the upper bound is ScStart and the lower bound is ScEnd.
          if (const auto *CStep = dyn_cast<SCEVConstant>(Step)) {
            if (CStep->getValue()->isNegative())
              std::swap(ScStart, ScEnd);
          } else {
            // fallback case: the step is not constant, but we can still
            // get the upper and lower bounds of the interval by using min/max expressions.
            ScStart = SE->getUMinExpr(ScStart, ScEnd);
            ScEnd = SE->getUMaxExpr(AR->getStart(), ScEnd);
          }
        }
        else{
          // could not compute
          continue;
        }
      }

      checks.emplace_back(InsertPt, Oper.OpType, ScStart, ScEnd, Oper.getInsn(), L, invariant, Addr, false);

    } // end of is_in_loop

  } // end of for all memory operations

  // after the SCEV constraints are generated, insert the checks 
  // (avoids our control flow insertions messing with SCEV)

  for (auto Range : checks) {

    // first we try to expand the SCEVs into IR
    // such that if that fails, we do not have to undo the inserted conditions
    // track the instruction before the insertion point, on which we will
    // split if the expansion succeeds

    if(optimized.find(Range.MemAccess) != optimized.end()){
      // post-metadata share check
      continue;
    }

    if(Range.reuseMeta){
      if(!Range.invariant){
        bool reuse_from_dom = false;

        // make sure the candidate is not a candidate for check merging, because that is more efficient
        bool skip_because_merge_cand = false;
        for (auto checkmerge : CheckMergers){
          if(checkmerge.TopTarget.getInsn() == Range.MemAccess){
            skip_because_merge_cand = true;
            break;
          }
          for(auto merge_cand : checkmerge.ContainedOps){
            if(merge_cand.getInsn() == Range.MemAccess){
              skip_because_merge_cand = true;
              break;
            }
          }
          if(skip_because_merge_cand) break; // out of the CheckMergers loop
        }

        // skip
        if(reuse_from_dom || skip_because_merge_cand) continue; 
      }

      // set up a check that reuses cached metadata inside the loop
      if(LoopPtrReuseMetadata(F, Range, CheckMergers, FuncMetaMergers, optimized)){
        optimized.insert(Range.MemAccess);
      }
      continue;
    }

    Value *StartValue = nullptr;
    Value *EndValue = nullptr;
    if(Range.invariant){
      StartValue = Range.Ptr;
    }
    else{
      SCEVExpander Expander(*SE, DL, "dummy_expander");
      StartValue = Expander.getRelatedExistingExpansion(Range.ScStart, Range.InsertPt, const_cast<Loop*>(Range.L));
      if (!StartValue && Expander.isSafeToExpandAt(Range.ScStart, Range.InsertPt))
        StartValue = Expander.expandCodeFor(Range.ScStart, nullptr, Range.InsertPt);
      
      if(!StartValue){
        // StartValue could not be expanded into valid IR
        continue;
      }

      EndValue = Expander.getRelatedExistingExpansion(Range.ScEnd, Range.InsertPt, const_cast<Loop*>(Range.L));
      if (!EndValue && Expander.isSafeToExpandAt(Range.ScEnd, Range.InsertPt))
        EndValue = Expander.expandCodeFor(Range.ScEnd, nullptr, Range.InsertPt);
      
      if(!EndValue){
        // EndValue could not be expanded into valid IR
        continue;
      }

    }

    Instruction *CheckInsertPt = Range.InsertPt; // default

    // insert the check (either in preheader or in exit block)
    if(Range.invariant){
      InsertCheck(*CheckInsertPt, *StartValue, true, Range.ptrType);
    }
    else{
      InsertCheckRange(*CheckInsertPt, StartValue, EndValue, Range.ptrType);
    }

    // remove the load/store in the loop from to-be-checked list
    optimized.insert(Range.MemAccess);
  }

	SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}
}

bool SafeStack::MoveAddrUp(Instruction *InsertPt, Instruction *Addr/*, PostDominatorTree &PDT*/){

  // NOTE: moving the address up is safe for check merges because the GEP
  // has to be a MustAlias (which would be false if the ptr could change)
  // and the GEP-offset is constant (so no data dependency possible)

#if 0
  for (Value *Op : Addr->operands()) {
    if(!DT.dominates(Op, InsertPt)){
      return false;
    }
  }
  if(!DT.dominates(Addr, InsertPt)){
    Addr->moveBefore(InsertPt);
  }
#else
  // is the target known at the insertion point?
  if(!DT.dominates(Addr, InsertPt)){
    // if not, we have to move the instruction, and check if the operands are reachable
    for (Value *Op : Addr->operands()) {
      // XXX: is there any relevant case where the operand is not an instruction,
      // and it would NOT be available at an earlier location?

      // PHINodes seem to cause problems in loops, do we need to check all incoming values
      // are movable...? can we even do this if it comes from a phinode?
      // i think phinodes may be unmovable candidates?
      // if(PHINode* phi = dyn_cast<PHINode>(Op)){
      //  return false;
      // }

      if(Instruction *IOp = dyn_cast<Instruction>(Op)){
        if(!MoveAddrUp(InsertPt, IOp/*, PDT*/)) return false;
      }
    }

    // I think we need to check whether there may be changes to one of the operands
    // in the instructions between the insertion point

    // check whether such a move would be valid, to avoid e.g. reading old data
    // if(!isSafeToMoveBefore(*Addr, *InsertPt, DT, &PDT, DI)){
      // return false;
    // }
    
    Addr->moveBefore(InsertPt);
  }
#endif
  return true;
}

bool SafeStack::IsAvailableToMove(Instruction *Ptr, Instruction *InsertPt, SmallPtrSetImpl<Value*> &Visited){
  // the Ptr itself currently likely does not dominate the InsertPt,
  // but if all its operands do dominate the insertpt, we can move the Ptr up
  // (this is quite conservative but seems safe)

  // avoid loops
  // if(Visited.count(Ptr)){
  //   return true;
  // }

  Visited.insert(Ptr);

  for (Value *Op : Ptr->operands()) {
    if(!isa<Instruction>(Op) && !isa<Constant>(Op) && !isa<Argument>(Op)){
      // cannot guarantee dominance
      return false;
    }

    if(!DT.dominates(Op, InsertPt)){
      return false;
    }

    // XXX: can we move up through phinodes?
    if(isa<PHINode>(Op)){
      return false;
    }

    if(isa<GlobalVariable>(Op) || isa<Argument>(Op)){
      // there may be data dependencies when moving these ptrs up earlier
      return false;
    }

    // see if the operands of the operand are available
    if(Instruction *IOp = dyn_cast<Instruction>(Op)){
      if(!IsAvailableToMove(IOp, InsertPt, Visited)){
        return false;
      }
    }
    else if(!isa<Constant>(Op)){
      // if its not an instruction and not a constant, there is a possibility
      // that there are data dependencies on the argument (changes between new and old position)
      // e.g.: globals, func args, ...
      // so conservatively skip
      return false;
    }
  }

  return true;
}


void SafeStack::MinMaxMergingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<MinMaxPair, 16> &MinMaxPairs, SmallVector<MinMaxChain, 16> &MinMaxChains, AliasAnalysis *AA, ScalarEvolution *SE){
  std::set<Instruction *> optimized;

#if 0 // disable chains: doesnt seem to help
  // first try chains
  for(MinMaxChain chain : MinMaxChains){
    if(!isOperInVector(chain.First, OperandsToInstrument)) continue;
    if(optimized.find(chain.First.getInsn()) != optimized.end()) continue;

    if(!isOperInVector(chain.Last, OperandsToInstrument)) continue;
    if(optimized.find(chain.Last.getInsn()) != optimized.end()) continue;

    bool give_up = false;
    for(auto contained : chain.ContainedOps){
      if(!isOperInVector(contained, OperandsToInstrument) || optimized.find(contained.getInsn()) != optimized.end()) {
        give_up = true;
        break;
      }
    }
    if(give_up) continue;

    // Check everything on the Last
    Instruction *InsertPt = chain.Last.getInsn();

    IRBuilder<> builder(InsertPt);

    // now find the min and max ptrs: start with the first and last, and then try all contained ones
    // (order should not really matter)
    Value *gmin = builder.CreateBinaryIntrinsic(Intrinsic::umin, chain.First.getPtr(), chain.Last.getPtr());
    Value *gmax = builder.CreateBinaryIntrinsic(Intrinsic::umax, chain.First.getPtr(), chain.Last.getPtr());

    optimized.insert(chain.First.getInsn());
    optimized.insert(chain.Last.getInsn());

    for(auto contained : chain.ContainedOps){
      gmin = builder.CreateBinaryIntrinsic(Intrinsic::umin, gmin, contained.getPtr());
      gmax = builder.CreateBinaryIntrinsic(Intrinsic::umax, gmax, contained.getPtr());
      optimized.insert(contained.getInsn());
    }

    InsertCheckRange(*InsertPt, gmin, gmax, chain.First.OpType);

  }
#endif

  // then pairs
  for(MinMaxPair minmax : MinMaxPairs){
    // see if this access was not already instrumented (e.g., by loop optimization)
    if(!isOperInVector(minmax.First, OperandsToInstrument)) continue;
    if(!isOperInVector(minmax.Second, OperandsToInstrument)) continue;

    // or by another minmax pair
    if (optimized.find(minmax.First.getInsn()) != optimized.end()) continue;
    if (optimized.find(minmax.Second.getInsn()) != optimized.end()) continue;

    // first try to process all pairs that have a 'load' as first instruction,
    // because then we can delay the check to the second operation 
    // (no risk of a store corrupting metadata)

    // for the first access, see if that is a load
    if(isa<LoadInst>(minmax.First.getInsn())){
      // we can check on the second access
      Instruction *InsertPt = minmax.Second.getInsn();

      IRBuilder<> builder(InsertPt);
      Value *gmin = builder.CreateBinaryIntrinsic(Intrinsic::umin, minmax.First.getPtr(), minmax.Second.getPtr());
      Value *gmax = builder.CreateBinaryIntrinsic(Intrinsic::umax, minmax.First.getPtr(), minmax.Second.getPtr());

      /*
        these umin/umax result in:
        cmp    r15,r13
        mov    rcx,r13
        mov    rdx,r13
        cmova  rdx,r15
        cmovb  rcx,r15
      */

      // First OpType == Second OpType
      InsertCheckRange(*InsertPt, gmin, gmax, minmax.First.OpType);

      optimized.insert(minmax.First.getInsn());
      optimized.insert(minmax.Second.getInsn());
    }
  }

  // for the remaining pairs: attempt to merge store-load/store pairs
  for(MinMaxPair minmax : MinMaxPairs){
    // see if this access was not already instrumented (e.g., by loop optimization)
    if(!isOperInVector(minmax.First, OperandsToInstrument)) continue;
    if(!isOperInVector(minmax.Second, OperandsToInstrument)) continue;

    // or by another minmax pair
    if (optimized.find(minmax.First.getInsn()) != optimized.end()) continue;
    if (optimized.find(minmax.Second.getInsn()) != optimized.end()) continue;

    // for the first access, see if that is a store
    if(isa<StoreInst>(minmax.First.getInsn())){
      // for stores we have to check on the first access
      Instruction *InsertPt = minmax.First.getInsn();

      // very conservatively see if the second ptr is available at the first ptr insertpt
      // the problem is that these non-constant GEPs can depend on variables
      // that are modified between the first and second check
      // we could move the ptr instruction up, but only if we can prove the data does not change
      if(DT.dominates(minmax.Second.getPtr(), InsertPt)){

          IRBuilder<> builder(InsertPt);
          Value *gmin = builder.CreateBinaryIntrinsic(Intrinsic::umin, minmax.First.getPtr(), minmax.Second.getPtr());
          Value *gmax = builder.CreateBinaryIntrinsic(Intrinsic::umax, minmax.First.getPtr(), minmax.Second.getPtr());

          // First OpType == Second OpType
          InsertCheckRange(*InsertPt, gmin, gmax, minmax.First.OpType);

          optimized.insert(minmax.First.getInsn());
          optimized.insert(minmax.Second.getInsn());
      }
    }
  }

	SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}
}


void SafeStack::NegPosMergingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<NegPosPair, 32> &NegPosPairs, AliasAnalysis *AA, ScalarEvolution *SE){
  std::set<Instruction *> optimized; 

  for(NegPosPair negpos : NegPosPairs){
    // see if this access was not already instrumented (e.g., by loop optimization)
    if(!isOperInVector(negpos.Pos, OperandsToInstrument)) continue;
    if(!isOperInVector(negpos.Neg, OperandsToInstrument)) continue;

    // or by another negpos pair
    if (optimized.find(negpos.Pos.getInsn()) != optimized.end()) continue;
    if (optimized.find(negpos.Neg.getInsn()) != optimized.end()) continue;


    // we need to be careful, because the GEP offsets are non-constant,
    // so we do not want to mess up data dependencies by moving instructions
    // if the first access is a store operation, the metadata could get corrupted if we delay the check to the second access
    // since we have postdominance, we can delay the check to the second access if the first is a load
    // the range check we insert is always [neg,pos]

    if(negpos.neg_first){
      // for the first access, see if that is a store
      if(isa<StoreInst>(negpos.Neg.getInsn())){
        // skip
      }
      else{
        // we can check on the second access
        Instruction *InsertPt = negpos.Pos.getInsn();
        // Sanity check: Neg ptr should be available at the Pos insertpt
        //if(DT.dominates(negpos.Neg.getPtr(), InsertPt))
        { // this should always be true given dominance

          InsertCheckRange(*InsertPt, negpos.Neg.getPtr(), negpos.Pos.getPtr(), negpos.Pos.OpType);

          optimized.insert(negpos.Neg.getInsn());
          optimized.insert(negpos.Pos.getInsn());
        }
      }
    }
    else{ // pos first
      // for the first access, see if that is a store
      if(isa<StoreInst>(negpos.Pos.getInsn())){
        // skip
      }
      else{
        // we can check on the second access
        Instruction *InsertPt = negpos.Neg.getInsn();
        // Sanity check: Pos ptr should be available at the Neg insertpt
        //if(DT.dominates(negpos.Pos.getPtr(), InsertPt))
        { // this should always be true given dominance
          
          InsertCheckRange(*InsertPt, negpos.Neg.getPtr(), negpos.Pos.getPtr(), negpos.Pos.OpType);

          optimized.insert(negpos.Neg.getInsn());
          optimized.insert(negpos.Pos.getInsn());
        }
      }
    }
  }

	SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}
}

void SafeStack::BlockSharingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &CheckMergers, AliasAnalysis *AA, ScalarEvolution *SE) {

  std::set<Instruction *> optimized; 

  // for every check top instruction, get the smallest and largest address
  // check if they dominate the insertion point, otherwise we have to move them (hopefully fine with checks in between...)
  // then insert a range check on the lower,upper bound

  for(auto merge : CheckMergers){

    // see if this access was not already instrumented (e.g., by loop optimization)
    if(!isOperInVector(merge.TopTarget, OperandsToInstrument)) continue;

    InterestingMemoryOperand Oper = merge.TopTarget;
    Instruction *InsertPt = Oper.getInsn();
    Type *AccessTy = merge.Biggest.Oper.OpType;

    Instruction *LowAddr = merge.Smallest.Addr;
    Instruction *HighAddr = merge.Biggest.Addr;

    // make sure the addresses and its dependencies are available at the insertion point

    if(!MoveAddrUp(InsertPt, LowAddr/*, PDT*/)) {
      continue;
    }
    if(!MoveAddrUp(InsertPt, HighAddr/*, PDT*/)) {
      continue;
    }

    InsertCheckRange(*InsertPt, LowAddr, HighAddr, AccessTy);
    optimized.insert(Oper.getInsn());

    for(InterestingMemoryOperand covered : merge.ContainedOps){
      optimized.insert(covered.getInsn());
    }

  }

	SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}
}

__attribute__((unused))
void SafeStack::FindFunctionMetaSharingCandidates(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &MetaReuse, AliasAnalysis *AA, ScalarEvolution *SE) {
  
  // we can re-use metadata for accesses on the same base with a larger offset
  // it is not sufficient to know the offset is positive, because the GEP base could be OOB

  DenseMap<OffsetSameBase*, SmallVector<OffsetSameBase, 16>> CheckCandidates;

  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {

    // we need to work with instructions to ensure we can move them if needed
    Instruction *IAddr = dyn_cast<Instruction>(Oper.getPtr());
    if(!IAddr) continue;

    if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Oper.getPtr()->stripPointerCasts())){

      // no GEPs from our instrumentation
      if(Instruction *IGEPPtr = dyn_cast<Instruction>(GEP->getPointerOperand())){
        if(IGEPPtr->hasMetadata("swiftsan")) continue;
      }

      if(!GEP->hasAllConstantIndices()) continue;

      // create a 'key' entry for this GEP
      APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0, true);
      if(!GEP->accumulateConstantOffset(DL, GEPOffset)) continue;

      OffsetSameBase *OSBThis = new OffsetSameBase(Oper, IAddr, GEPOffset, false);
      
      // candidates
      for(InterestingMemoryOperand &Other : OperandsToInstrument){       
        Instruction *access = Oper.getInsn();
        if(access == Other.getInsn()) continue; // itself

        Instruction *OtherIAddr = dyn_cast<Instruction>(Other.getPtr());
        if(!OtherIAddr) continue;

        // the other access should also come from a GEP
        GetElementPtrInst *OtherGEP = dyn_cast<GetElementPtrInst>(Other.getPtr()->stripPointerCasts());
        if(!OtherGEP) continue;

        if(!OtherGEP->hasAllConstantIndices()) continue;

        // no GEPs from our instrumentation
        if(Instruction *IGEPOtherPtr = dyn_cast<Instruction>(OtherGEP->getPointerOperand())){
          if(IGEPOtherPtr->hasMetadata("swiftsan")) continue;
        }

        // same base
        if(!AA->isMustAlias(GEP->getPointerOperand(), OtherGEP->getPointerOperand()))
          continue;

        // dominate the other access
        if(!DT.dominates(access, Other.getInsn()))
          continue;

        // for every Other, if the GEP offset is provably larger or smaller, we can merge the check
        // this then needs to accumulate for the 'highest' and 'lowest' total of the merge
        // APInt OtherOffset(DL.getIndexTypeSizeInBits(OtherGEP->getType()), 0);
        APInt OtherOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0, true);

        // calculate the accumulated GEP offset
        if(!OtherGEP->accumulateConstantOffset(DL, OtherOffset)) continue;

        bool otherIsSmaller;
        if(OtherOffset.sgt(GEPOffset)){
          otherIsSmaller = false;         
        }
        else{
          // for metadata we only look for larger offsets
          continue;
        }

        // Track all the candidates.
        CheckCandidates[OSBThis].emplace_back(Other, OtherIAddr, OtherOffset, otherIsSmaller);
      }
    }
  }

  for(auto entry : CheckCandidates){
    OffsetSameBase *ThisOSB = entry.first;
    InterestingMemoryOperand KeyOper = ThisOSB->Oper;
    bool covered = false;

    for(auto other : CheckCandidates){
      if(KeyOper.getInsn() == other.first->Oper.getInsn()) continue; // itself

      for(auto osb : other.second){
        if(KeyOper.getInsn() == osb.Oper.getInsn()){
          covered = true;
          break;
        }
      }
    }

    if(covered) continue;

    // the smallest/biggest do not matter for metadata reuse (redundant)
    OffsetSameBase Smallest(*ThisOSB);
    OffsetSameBase Biggest(*ThisOSB);

    SmallVector<InterestingMemoryOperand, 16> ContainedOps;

    // all that can re-use (i.e. bigger)
    for(auto osb : entry.second){
      ContainedOps.push_back(osb.Oper);
    }

    MetaReuse.emplace_back(KeyOper, ContainedOps, Smallest, Biggest);
  }

  // clean up
  for(auto entry : CheckCandidates){
    OffsetSameBase *ThisOSB = entry.first;
    delete ThisOSB;
  }

}

bool GEPOffsetIsPositive(GetElementPtrInst *GEP, ScalarEvolution *SE){
  for (auto &Op : GEP->indices()) {
    auto Range = SE->getSCEV(Op.get());
    // not non-negative operand == not provably positive
    if (!SE->getSignedRangeMin(Range).isNonNegative()){
      return false;
    }
  }
  // positive otherwise
  return true;
}

bool GEPOffsetIsNegative(GetElementPtrInst *GEP, ScalarEvolution *SE){
  for (auto &Op : GEP->indices()) {
    auto Range = SE->getSCEV(Op.get());
    // non-negative operand == not provably negative
    if (!SE->getSignedRangeMin(Range).isNegative()){
      return false;
    }
  }
  // positive otherwise
  return true;
}


void SafeStack::FindMinMaxPairs(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<MinMaxPair, 16> &MinMaxPairs, SmallVector<MinMaxChain, 16> &MinMaxChains, AliasAnalysis *AA, ScalarEvolution *SE){

  // evaluate the min and max address of two accesses on the same base,
  // after which we can insert a merged range check on [min,max]

  auto PDT = PostDominatorTree(F);
  
  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {

    // we need to work with instructions to ensure we can move them if needed
    Instruction *IAddr = dyn_cast<Instruction>(Oper.getPtr());
    if(!IAddr) continue;

    if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Oper.getPtr()->stripPointerCasts())){

      // no GEPs from our instrumentation
      if(Instruction *IGEPPtr = dyn_cast<Instruction>(GEP->getPointerOperand())){
        if(IGEPPtr->hasMetadata("swiftsan")) continue;
      }
      
      // candidates
      for(InterestingMemoryOperand &Other : OperandsToInstrument){       
        Instruction *access = Oper.getInsn();
        if(access == Other.getInsn()) continue; // itself

        Instruction *OtherIAddr = dyn_cast<Instruction>(Other.getPtr());
        if(!OtherIAddr) continue;

        // the other access should also come from a GEP
        GetElementPtrInst *OtherGEP = dyn_cast<GetElementPtrInst>(Other.getPtr()->stripPointerCasts());
        if(!OtherGEP) continue;

        // no GEPs from our instrumentation
        if(Instruction *IGEPOtherPtr = dyn_cast<Instruction>(OtherGEP->getPointerOperand())){
          if(IGEPOtherPtr->hasMetadata("swiftsan")) continue;
        }

        // same base
        if(!AA->isMustAlias(GEP->getPointerOperand(), OtherGEP->getPointerOperand()))
          continue;

        // dominate the other access
        if(!DT.dominates(access, Other.getInsn()))
          continue;

        // the later access needs to postdominate the earlier
        if(!PDT.dominates(Other.getInsn(), access))
          continue;

        // TODO: if the pointers are in a loop and loop invariant,
        // we can calculate the min,max once in the preheader (assuming unconditional)

        TypeSize szFirst = DL.getTypeStoreSize(Oper.OpType);
        TypeSize szSecond = DL.getTypeStoreSize(Other.OpType);

        // only allow min-max pairs with the same access size, 
        // such that we know which access size to use on the check
        // otherwise, we have to use the minimum, meaning partial overflows
        // can go undetected.
        if(szFirst == szSecond){
          MinMaxPairs.emplace_back(Oper, Other);
        }
      }
    }
  }

}

void SafeStack::FindNegPosPairs(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<NegPosPair, 32> &NegPosPairs, AliasAnalysis *AA, ScalarEvolution *SE){

  // in the entire function find dom-postdom pairs, where one is negative and one is positive
  // we collect all possible pairs and then simply try to merge them unless the check is already optimized
  // since we do not know if a merge can succeed beforehand
  auto PDT = PostDominatorTree(F);

  for (InterestingMemoryOperand &Oper : OperandsToInstrument) {

    // we need to work with instructions to ensure we can move them if needed
    Instruction *IAddr = dyn_cast<Instruction>(Oper.getPtr());
    if(!IAddr) continue;

    if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Oper.getPtr()->stripPointerCasts())){

      // no GEPs from our instrumentation
      if(Instruction *IGEPPtr = dyn_cast<Instruction>(GEP->getPointerOperand())){
        if(IGEPPtr->hasMetadata("swiftsan")) continue;
      }
      
      // candidates
      for(InterestingMemoryOperand &Other : OperandsToInstrument){       
        Instruction *access = Oper.getInsn();
        if(access == Other.getInsn()) continue; // itself

        Instruction *OtherIAddr = dyn_cast<Instruction>(Other.getPtr());
        if(!OtherIAddr) continue;

        // the other access should also come from a GEP
        GetElementPtrInst *OtherGEP = dyn_cast<GetElementPtrInst>(Other.getPtr()->stripPointerCasts());
        if(!OtherGEP) continue;

        // no GEPs from our instrumentation
        if(Instruction *IGEPOtherPtr = dyn_cast<Instruction>(OtherGEP->getPointerOperand())){
          if(IGEPOtherPtr->hasMetadata("swiftsan")) continue;
        }

        // same base
        if(!AA->isMustAlias(GEP->getPointerOperand(), OtherGEP->getPointerOperand()))
          continue;

        // dominate the other access
        if(!DT.dominates(access, Other.getInsn()))
          continue;

        // the later access needs to postdominate the earlier
        if(!PDT.dominates(Other.getInsn(), access))
          continue;

        // Now, the GEP and OtherGEP form a pair if one is negative and the other is positive
        if(GEPOffsetIsNegative(GEP, SE) && GEPOffsetIsPositive(OtherGEP, SE)){
          NegPosPairs.emplace_back(Oper, Other, true); // neg, pos, neg dom pos
        }
        else if(GEPOffsetIsNegative(OtherGEP, SE) && GEPOffsetIsPositive(GEP, SE)){
          NegPosPairs.emplace_back(Other, Oper, false); // neg, pos, pos dom neg
        }
        else{
          // not a pair, keep searching.
        }
      }
    }
  }
}

void SafeStack::FindBlockCheckSharingCandidates(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &CheckMergers, AliasAnalysis *AA, ScalarEvolution *SE) {
  
  // gather all the basicblocks (before we start modifying blocks)
  SmallSetVector<BasicBlock *, 8> blocks;
  for (BasicBlock &BB : F) {
    blocks.insert(&BB);
  }

  for (BasicBlock *BB : blocks) {
    SmallVector<InterestingMemoryOperand, 16> BlockOpers;
    for (InterestingMemoryOperand &Oper : OperandsToInstrument) {
      if(Oper.getInsn()->getParent() == BB){
        BlockOpers.push_back(Oper);
      }
    }

    DenseMap<OffsetSameBase*, SmallVector<OffsetSameBase, 16>> CheckCandidates;

    for(InterestingMemoryOperand &Oper : BlockOpers){

      // we need to work with instructions to ensure we can move them if needed
      Instruction *IAddr = dyn_cast<Instruction>(Oper.getPtr());
      if(!IAddr) continue;

      if(GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Oper.getPtr()->stripPointerCasts())){
        // check the other load/stores in the Block
        // they should access the same base, and have a higher GEP offset
        // and we need to ensure domination (pick the top one)

        // here we only reason about constant offsets, because we need to prove they are larger/smaller than others

        // no GEPs from our instrumentation
        if(Instruction *IGEPPtr = dyn_cast<Instruction>(GEP->getPointerOperand())){
          if(IGEPPtr->hasMetadata("swiftsan")) continue;
        }

        if(!GEP->hasAllConstantIndices()) continue;

        // create a 'key' entry for this GEP
        APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0, true);
        if(!GEP->accumulateConstantOffset(DL, GEPOffset)) continue;

        OffsetSameBase *OSBThis = new OffsetSameBase(Oper, IAddr, GEPOffset, false);

        // if we can confirm Ptr is stack/global memory, then potential dealloc calls are not relevant
        bool stack_or_glob = false;
        if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(GEP))) {
          stack_or_glob = true;
        }
        if (isa<AllocaInst>(getUnderlyingObject(GEP))) {
          stack_or_glob = true;
        }
        // unsafestack is also stack
        Value *base = nullptr;
        Constant *og_size = nullptr;
        SmallPtrSet<Value *, 4> Visited;
        AccumulateToUnsafeStackAlloca(GEP, Visited, &base, &og_size);
        if(base) {
          stack_or_glob = true;
        }
       
        // candidates
        for(InterestingMemoryOperand &Other : BlockOpers){       
          Instruction *access = Oper.getInsn();
          if(access == Other.getInsn()) continue; // itself

          Instruction *OtherIAddr = dyn_cast<Instruction>(Other.getPtr());
          if(!OtherIAddr) continue;

          // the other access should also come from a GEP
          GetElementPtrInst *OtherGEP = dyn_cast<GetElementPtrInst>(Other.getPtr()->stripPointerCasts());
          if(!OtherGEP) continue;
  
          if(!OtherGEP->hasAllConstantIndices()) continue;

          // no GEPs from our instrumentation
          if(Instruction *IGEPOtherPtr = dyn_cast<Instruction>(OtherGEP->getPointerOperand())){
            if(IGEPOtherPtr->hasMetadata("swiftsan")) continue;
          }

          // same base
          if(!AA->isMustAlias(GEP->getPointerOperand(), OtherGEP->getPointerOperand()))
            continue;

          // dominate the other access
          if(!DT.dominates(access, Other.getInsn()))
            continue;

          if(!stack_or_glob){
            // check that there are no calls between the top instruction and this candidate
            bool call_inbetween = false;
            Instruction *Next = access;
            while(Next != Other.getInsn()){
              if (CallBase *CB = dyn_cast<CallBase>(Next)) {
                if(CB->isDebugOrPseudoInst() || CB->isLifetimeStartOrEnd()){
                  Next = Next->getNextNode();
                  continue;
                }
                // less conservative: a call inbetween of which the arg is an alias of the ptr
                for (Value *arg : CB->args()){
#if 0 // ASan-- arg == ptr without aliasing
                  // ASan--: arg == ptr
                  if(arg == GEP->getPointerOperand() || arg == GEP){
                    call_inbetween = true;
                    break;
                  }
#else
                  AliasResult a = AA->alias(GEP->getPointerOperand(), arg);
                  if(a == AliasResult::MayAlias || a == AliasResult::MustAlias ){
                    call_inbetween = true;
                    break;
                  }
#endif
                }
                if(call_inbetween) break;
              }

              Next = Next->getNextNode();
              if(!Next) break;
            }
            if(call_inbetween) continue;
          }

          // for every Other, if the GEP offset is provably larger or smaller, we can merge the check
          // this then needs to accumulate for the 'highest' and 'lowest' total of the merge
          // APInt OtherOffset(DL.getIndexTypeSizeInBits(OtherGEP->getType()), 0);
          APInt OtherOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0, true);

          // calculate the accumulated GEP offset
          if(!OtherGEP->accumulateConstantOffset(DL, OtherOffset)) continue;

          bool otherIsSmaller;
          if(OtherOffset.sgt(GEPOffset)){
            otherIsSmaller = false;         
          }
          else if(OtherOffset.slt(GEPOffset)){
            otherIsSmaller = true;
          }
          else{
            // equal? the gep shouldve been optimized out probably
            // and otherwise I assume 'recurring check' optimization handles this
            continue;
          }

          // Track all the candidates.
          // First determine if the entry is a candidate in another, if that is the case, we can skip it (transitivity)
          // After finding the 'top' entry, we should try to create a class with the 'min' and 'max' offsets
          // and move these to the top in one check
          CheckCandidates[OSBThis].emplace_back(Other, OtherIAddr, OtherOffset, otherIsSmaller);

          // Instruction -> AllSmallerThan, AllBiggerThan
          // if Instruction is in another Smaller or Bigger Than, delete the entry

        }
      }
    }

    for(auto entry : CheckCandidates){
      OffsetSameBase *ThisOSB = entry.first;
      InterestingMemoryOperand KeyOper = ThisOSB->Oper;
      bool covered = false;

      for(auto other : CheckCandidates){
        if(KeyOper.getInsn() == other.first->Oper.getInsn()) continue; // itself

        for(auto osb : other.second){
          if(KeyOper.getInsn() == osb.Oper.getInsn()){
            covered = true;
            break;
          }
        }
      }

      if(covered) continue;

      // we should start with smallest = GEP and largest = GEP
      // then check if osb.smaller, see if smaller than smallest, etc
      OffsetSameBase Smallest(*ThisOSB);
      OffsetSameBase Biggest(*ThisOSB);

      SmallVector<InterestingMemoryOperand, 16> ContainedOps;

      // loop through the offsets and find the smallest and biggest
      for(auto osb : entry.second){
        if(osb.smaller){
          if(osb.Offset.slt(Smallest.Offset)){
            Smallest = osb;
          }
        }
        else{ // bigger
          // include the access size to find the truly largest offset access 
          APInt CurrTotalOffset = Biggest.Offset + APInt(Biggest.Offset.getBitWidth(), DL.getTypeStoreSize(Biggest.Oper.OpType)-1);
          APInt OtherTotalOffset = osb.Offset + APInt(osb.Offset.getBitWidth(), DL.getTypeStoreSize(osb.Oper.OpType)-1);
          if(OtherTotalOffset.sgt(CurrTotalOffset)){
            Biggest = osb;
          }
        }
        ContainedOps.push_back(osb.Oper);
      }

      CheckMergers.emplace_back(KeyOper, ContainedOps, Smallest, Biggest);
    }

    // clean up
    for(auto entry : CheckCandidates){
      OffsetSameBase *ThisOSB = entry.first;
      delete ThisOSB;
    }
  }
}

__attribute__((unused))
void SafeStack::MetadataSharingOptimization(Function &F, SmallVector<InterestingMemoryOperand, 16> &OperandsToInstrument, SmallVector<CheckMergeCands, 16> &FuncMetaMergers, AliasAnalysis *AA, ScalarEvolution *SE) {

  // TODO: if the target already has a check, but there are candidates that do not
  // we need to find the tag,end-of-obj pair so we can still try to re-use.
  // this requires tracking all inserted checks so far

  std::set<Instruction *> optimized;

  for(auto share : FuncMetaMergers){

    // see if this access was not already instrumented (e.g., by loop optimization)
    if(!isOperInVector(share.TopTarget, OperandsToInstrument)) continue;


    int remaining_candidates = 0;

    // the sharing candidates: see if they do not already have a check
    for(InterestingMemoryOperand covered : share.ContainedOps){
      if(isOperInVector(covered, OperandsToInstrument)){
        remaining_candidates++;
      }
    }

    if(remaining_candidates > 0){
      // there are candidates for reusing metadata, lets insert propgation checks

      std::tuple<Value*, Value*> EndNTag;
      EndNTag = InsertCheck(*share.TopTarget.getInsn(), *share.TopTarget.getPtr(), true, share.TopTarget.OpType);
      optimized.insert(share.TopTarget.getInsn());

      for(InterestingMemoryOperand covered : share.ContainedOps){
        if(isOperInVector(covered, OperandsToInstrument)){
          InsertCheckMeta(*covered.getInsn(), *covered.getPtr(), true, covered.OpType, std::get<0>(EndNTag), std::get<1>(EndNTag), false);
          optimized.insert(covered.getInsn());
        }
      }

    }

  }

  SmallVector<InterestingMemoryOperand, 16> LOTempToInstrument(OperandsToInstrument);
	OperandsToInstrument.clear();

	for (auto item: LOTempToInstrument) {
		if (optimized.find(item.getInsn()) == optimized.end())
			OperandsToInstrument.push_back(item);
	}

}

void SafeStack::getInterestingMemoryOperands(Instruction &I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {

  if (I.hasMetadata("nosanitize") || I.hasMetadata("swiftsan")) {
    return;
  }

  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
    Interesting.emplace_back(&I, LI->getPointerOperandIndex(), false,
                                      LI->getType(), LI->getAlign());
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
    Interesting.emplace_back(&I, SI->getPointerOperandIndex(), true,
                                      SI->getValueOperand()->getType(), SI->getAlign());
  }
  //TODO Missing atomic mem operations: AtomicRMWInst, AtomicCmpXchgInst, llvm.masked.load, llvm.masked.store
}

bool SafeStack::ChecksOnFunc(Function &F, ObjectSizeOffsetVisitor &ObjSizeVis) {

  // The load/store pointers to be instrumented
  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  // Temp vector to track only instrumenting every address once per BB
  SmallPtrSet<Value *, 16> TempsToInstrument;

  for (BasicBlock &BB : F) {
    TempsToInstrument.clear();

    for (Instruction &I : BB) {
      // load + store
      SmallVector<InterestingMemoryOperand, 1> InterestingOperands;
      getInterestingMemoryOperands(I, InterestingOperands);

      if (!InterestingOperands.empty()) {
        for (auto &Operand : InterestingOperands) {
          Value *Ptr = Operand.getPtr();
          if (!TempsToInstrument.insert(Ptr).second) {
            continue; // We've seen this temp in the current BB.
          }
          OperandsToInstrument.push_back(Operand);
        }
      }
    }
  }
  
  // --- Optimizations to reduce the number of checks ---
  // #1: Default ASan IsSafeAccess with SafeStack assumption
  // + ASAN--: Removing Unsatisfiable Checks
  OptStaticallySafeAccesses(ObjSizeVis, OperandsToInstrument); 

  // #2: Reverse the order of safe stack allocations: all remaining alloca uses are safe
  OptStaticallySafeAllocas(OperandsToInstrument);

  // #2-B: Unsafe stack accesses with constant offsets can be proven safe 
  OptStaticallySafeUnsafeStack(OperandsToInstrument);

  // #3: ASAN--: Removing Recurring Checks
  sequentialExecuteOptimizationPostDom(F, OperandsToInstrument, AA);
  sequentialExecuteOptimization(F, OperandsToInstrument, AA);

  // #4: ASAN--: Optimizing Neighbor Checks (remove only, no merge)
  sequentialExecuteOptimizationBoost(F, OperandsToInstrument);


  // Classify into two clusters: ptr variant and invariant in loops
  SmallVector<InterestingMemoryOperand, 16> invariants;
  ClassifyLoopAccesses(F, OperandsToInstrument, invariants, &LI, &SE);

  // Gather BasicBlock-level metadata sharing candidates
  // but Block-level is useful for Loops: sharing within a BB implies loop body compatibility
  // otherwise we may need Loop-level candidates, because Function-level is too high
  SmallVector<CheckMergeCands, 16> CheckMergers;
  FindBlockCheckSharingCandidates(F, OperandsToInstrument, CheckMergers, AA, &SE);

  // Find [neg,pos] pairs on the same base to merge the two checks into one
  // SCEV cannot statically compare pointers, but if we know one is positive and one is negative
  // then we can perform a range check
  // Find all possible pairs and try to merge them, since this can fail due to data dependencies
  SmallVector<NegPosPair, 32> NegPosPairs;
  FindNegPosPairs(F, OperandsToInstrument, NegPosPairs, AA, &SE);

  // Find [min,max] pairs that require runtime evaluation for which one is greater
  SmallVector<MinMaxPair, 16> MinMaxPairs;
  SmallVector<MinMaxChain, 16> MinMaxChains;
  FindMinMaxPairs(F, OperandsToInstrument, MinMaxPairs, MinMaxChains, AA, &SE);

  // for Function-level, we currently showcase metadata sharing, although check merging is
  // also possible if post-dom chains are considered (more complex)
  SmallVector<CheckMergeCands, 16> FuncMetaMergers;
  if (ClSubsanMetadataSharing) {
    FindFunctionMetaSharingCandidates(F, OperandsToInstrument, FuncMetaMergers, AA, &SE);
  }

  // TODO: there is possible interplay between optimizations:
  // i.e.: cache the metadata of a check that covers a larger range
  // e.g.: a loop allows the metadata of a ptr to be cached,
  // but this ptr is part of a check-merging condition, which now favors check-merging
  // while metadata caching could also be applied there, still

  // #5: BaseBounds Loop Optimizations
  LoopOptimization(F, OperandsToInstrument, invariants, CheckMergers, FuncMetaMergers, &LI, &SE);

  // #6: Block-based sharing optimization (Const Offset Merging)
  BlockSharingOptimization(F, OperandsToInstrument, CheckMergers, AA, &SE);

  // #7: Function-level Neg/Pos check merging optimization
  NegPosMergingOptimization(F, OperandsToInstrument, NegPosPairs, AA, &SE);

  MinMaxMergingOptimization(F, OperandsToInstrument, MinMaxPairs, MinMaxChains, AA, &SE);

  // #8: Metadata sharing optimization
  if (ClSubsanMetadataSharing) {
    MetadataSharingOptimization(F, OperandsToInstrument, FuncMetaMergers, AA, &SE);
  }

  // Instrument remaining load/store targets with checks
  for (auto &MO : OperandsToInstrument) {
    InsertCheck(*MO.getInsn(), *MO.getPtr(), MO.IsWrite, MO.OpType);
  }

  return true;
}

void SafeStack::InstrumentCalls(Function &F) {
  if (isNoInstrument(&F)) return;

  Module *M = F.getParent();
  std::vector<CallInst *> calls;
  SmallVector<MemIntrinsic *, 16> intrins;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (I.hasMetadata("swiftsan")) continue;
      if (CallInst *C = llvm::dyn_cast<CallInst>(&I)) {
          calls.push_back(C);
      }
      if(MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&I)){
          intrins.push_back(MI);
      }
    }
  }
  
  // libc memory operations
  for (CallInst *CI : calls) {
    Function *callTarget = CI->getCalledFunction();
    // Indirect call.
    if (!callTarget)
      continue;

    // Filter out functions not intended to be intercepted.
    std::string name = callTarget->getName().str();
    if (!interceptors.contains(name))
      continue;

    std::string replName = interceptPrefix + name;
    FunctionCallee replacement = M->getOrInsertFunction(replName, callTarget->getFunctionType());

    if (ClSubsanMallocAlign) {
      if (Function *RF = dyn_cast<Function>(replacement.getCallee())) {
        if (name == "malloc" || name == "calloc" || name == "realloc") {
          RF->addRetAttr(Attribute::NoAlias);
          RF->addRetAttr(Attribute::getWithAlignment(F.getContext(), Align(16)));
        }
      }
    }

    {
      IRBuilder<> builder(CI);
      std::vector<Value *> args;
      for (Value *arg : CI->args())
        args.push_back(arg);
      CallInst *replacedCall = builder.CreateCall(replacement, args);
      assert(replacedCall);
      replacedCall->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
      CI->replaceAllUsesWith(replacedCall);
      CI->eraseFromParent();
    }
  }

  // llvm memory transfers (mem intrinsics)
  // at this point, any memory transfer/set on an alloca
  // has to be statically safe, because of safestack
  // for those cases, use dest/src-only checks, or skip the memset
  // similarly for skipped globals

  for (MemIntrinsic *MI : intrins) {
    IRBuilder<> IRB(MI);
    if (isa<MemTransferInst>(MI)) {

      FunctionCallee targetCall;
      Value *dst = MI->getOperand(0);
      Value *src = MI->getOperand(1);
      bool dst_safe = false;
      bool src_safe = false;
      if (isa<AllocaInst>(getUnderlyingObject(dst))) {
        dst_safe = true;
      }
      else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(dst))) {
        if(!shouldInstrumentGlobal(*GV)){
          dst_safe = true;
        }
      }
      if (isa<AllocaInst>(getUnderlyingObject(src))) {
        src_safe = true;
      }
      else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(src))) {
        if(!shouldInstrumentGlobal(*GV)){
          src_safe = true;
        }
      }
      if(src_safe && dst_safe){
        continue;
      }
      else if(src_safe){ // unsafe dst
        targetCall = isa<MemMoveInst>(MI) ? SwiftsanMemmoveDst : SwiftsanMemcpyDst;
      }
      else if(dst_safe){ // unsafe src
        targetCall = isa<MemMoveInst>(MI) ? SwiftsanMemmoveSrc : SwiftsanMemcpySrc;
      }
      else{
        // neither src/dest are provably safe, check both
        targetCall = isa<MemMoveInst>(MI) ? SwiftsanMemmove : SwiftsanMemcpy;
      }

      IRB.CreateCall(
          targetCall,
          {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
           IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
           IRB.CreateIntCast(MI->getOperand(2), IntPtrTy, false)});

    } else if (isa<MemSetInst>(MI)) {

      // statically safe
      if (isa<AllocaInst>(getUnderlyingObject(MI->getOperand(0)))) {
        continue;
      }
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(MI->getOperand(0)))) {
        if(!shouldInstrumentGlobal(*GV)){
          continue;
        }
      }

      IRB.CreateCall(
          SwiftsanMemset,
          {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
           IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
           IRB.CreateIntCast(MI->getOperand(2), IntPtrTy, false)});
    } else {
      llvm_unreachable("Neither MemSet nor MemTransfer?");
    }
    MI->eraseFromParent();
  }

}

void AccumulateToVarStart(Value *V, SmallPtrSetImpl<Value*> &Visited, bool *found){
  V = V->stripPointerCastsAndAliases();
  if (Visited.count(V))
    return;
  if(PHINode* phi = dyn_cast<PHINode>(V)){
    Visited.insert(phi);
    for(Value *Inc : phi->incoming_values()){
      return AccumulateToVarStart(Inc, Visited, found);
    }
  }
  else if(GetElementPtrInst* gep = dyn_cast<GetElementPtrInst>(V)){
    Visited.insert(gep);
    return AccumulateToVarStart(gep->getPointerOperand(), Visited, found);
  }
  else if(LoadInst* load = dyn_cast<LoadInst>(V)){
    Visited.insert(load);
    return AccumulateToVarStart(load->getPointerOperand(), Visited, found);
  }
  else if(CastInst *cast = dyn_cast<CastInst>(V)){
    Visited.insert(cast);
    return AccumulateToVarStart(cast->getOperand(0), Visited, found);
  }
  else if(AllocaInst* alloca = dyn_cast<AllocaInst>(V)){
    Visited.insert(alloca);
    // first check if the type of the alloca is va_list
    if(StructType *allocaStruct = dyn_cast<StructType>(alloca->getAllocatedType())){
      if(allocaStruct->hasName() && allocaStruct->getName() == "struct.__va_list"){
        *found = true;
        return;
      }
    }
    // otherwise, as fallthrough: check the uses of the alloca for va_start() call
    for (const Use &UI : V->uses()) {
      if(CallInst *CIuse = dyn_cast<CallInst>(UI.getUser())){
        Function *callTarget = CIuse->getCalledFunction();
        if (!callTarget) // Indirect call.
          continue;

        if(callTarget->getName() == "llvm.va_start"){
          *found = true;
          return;
        }
      }
    }
  }
  else{
    Visited.insert(V);
  }
}

void SafeStack::MarkAllVarArgLoads(Function &F) {
  for (Instruction &I : instructions(&F)) {
    if (LoadInst *Load = dyn_cast<LoadInst>(&I)) {
      SmallPtrSet<Value *, 4> Visited;
      bool var_arg_load = false;
      AccumulateToVarStart(Load, Visited, &var_arg_load);
      if (var_arg_load) {
        Load->setMetadata(F.getParent()->getMDKindID("swiftsan"),
                          llvm::MDNode::get(F.getContext(), std::nullopt));
      }
    }
  }
}

void SafeStack::MarkErrnoAccesses(Function &F) {
  // we search for the form 'errno = X'
  for (Instruction &I : instructions(&F)) {
    if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
      if (CallInst *CIptr = dyn_cast<CallInst>(Store->getPointerOperand())) {
        Function *callTarget = CIptr->getCalledFunction();
        if (!callTarget) // Indirect call.
          continue;
        if (callTarget->getName() == "__errno_location") {
          Store->setMetadata(F.getParent()->getMDKindID("swiftsan"),
                             llvm::MDNode::get(F.getContext(), std::nullopt));
        }
      }
    }
    else if (LoadInst *Load = dyn_cast<LoadInst>(&I)) {
      if (CallInst *CIptr = dyn_cast<CallInst>(Load->getPointerOperand())) {
        Function *callTarget = CIptr->getCalledFunction();
        if (!callTarget) // Indirect call.
          continue;
        if (callTarget->getName() == "__errno_location") {
          Load->setMetadata(F.getParent()->getMDKindID("swiftsan"),
                             llvm::MDNode::get(F.getContext(), std::nullopt));
        }
      }
    }
  }
}

bool SafeStack::run() {
  assert(F.hasFnAttribute(Attribute::SafeStack) &&
         "Can't run SafeStack on a function without the attribute");
  assert(!F.isDeclaration() && "Can't run SafeStack on a function declaration");

  // AttrBuilder AB(F.getParent()->getContext());
  // AB.addAttribute(Attribute::StackProtect);
  // AB.addAttribute(Attribute::StackProtectReq);
  // AB.addAttribute(Attribute::StackProtectStrong);
  // F.removeAttributes(AttributeList::FunctionIndex, AB);

  // Initial Steps: mark va_arg loads as no instrument 
  // we do this before va_list alloca info is gone through SafeStack
  // and before pointer tagging bloats the instructions
  // var args are pushed on the stack, but not through an alloca
  // so they are always uninstrumented
  // similarly for errno and argv: they are also uninstrumented
  // argv we move to the heap
  // errno and vararg accesses are statically safe anyway i think
  // environ/envp i think we cannot update, because setenv() in libc 
  // needs to remain compatible

  MarkAllVarArgLoads(F);
  MarkErrnoAccesses(F);

  // Testing for 620.omnetpp_s
  // MarkCTypeLoads(F);

  ++NumFunctions;

  SmallVector<AllocaInst *, 16> StaticAllocas;
  SmallVector<AllocaInst *, 4> DynamicAllocas;
  SmallVector<Argument *, 4> ByValArguments;
  SmallVector<Instruction *, 4> Returns;

  // Collect all points where stack gets unwound and needs to be restored
  // This is only necessary because the runtime (setjmp and unwind code) is
  // not aware of the unsafe stack and won't unwind/restore it properly.
  // To work around this problem without changing the runtime, we insert
  // instrumentation to restore the unsafe stack pointer when necessary.
  SmallVector<Instruction *, 4> StackRestorePoints;
  // Find all static and dynamic alloca instructions that must be moved to the
  // unsafe stack, all return instructions and stack restore points.
  // F.dump();
  findInsts(F, StaticAllocas, DynamicAllocas, ByValArguments, Returns,
            StackRestorePoints);

  bool noStackUsage = StaticAllocas.empty() && DynamicAllocas.empty() &&
      ByValArguments.empty() && StackRestorePoints.empty();
  
  if (!noStackUsage) {
    if (!StaticAllocas.empty() || !DynamicAllocas.empty() ||
        !ByValArguments.empty())
      ++NumUnsafeStackFunctions; // This function has the unsafe stack.

    if (!StackRestorePoints.empty())
      ++NumUnsafeStackRestorePointsFunctions;

    IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());
    // Calls must always have a debug location, or else inlining breaks. So
    // we explicitly set a artificial debug location here.
    if (DISubprogram *SP = F.getSubprogram())
      IRB.SetCurrentDebugLocation(
          DILocation::get(SP->getContext(), SP->getScopeLine(), 0, SP));
    if (SafeStackUsePointerAddress) {
      FunctionCallee Fn = F.getParent()->getOrInsertFunction(
          "__safestack_pointer_address", StackPtrTy->getPointerTo(0));
      Instruction *ICall = IRB.CreateCall(Fn);
      ICall->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
      UnsafeStackPtr = ICall;
    } else {
      UnsafeStackPtr = TL.getSafeStackPointerLocation(IRB);
      if (auto *TLSGV = dyn_cast<GlobalVariable>(UnsafeStackPtr)) {
        if (TLSGV->isThreadLocal() &&
            TLSGV->getThreadLocalMode() != GlobalVariable::LocalExecTLSModel)
          TLSGV->setThreadLocalMode(GlobalVariable::LocalExecTLSModel);
      }
    }

    AllocaInst *StackGuardSlot = nullptr;
    // // FIXME: implement weaker forms of stack protector.
    // if (F.hasFnAttribute(Attribute::StackProtect) ||
    //     F.hasFnAttribute(Attribute::StackProtectStrong) ||
    //     F.hasFnAttribute(Attribute::StackProtectReq)) {
    //   Value *StackGuard = getStackGuard(IRB, F);
    //   StackGuardSlot = IRB.CreateAlloca(StackPtrTy, nullptr);
    //   IRB.CreateStore(StackGuard, StackGuardSlot);

    //   for (Instruction *RI : Returns) {
    //     IRBuilder<> IRBRet(RI);
    //     checkStackGuard(IRBRet, F, *RI, StackGuardSlot, StackGuard);
    //   }
    // }

    // Handle dynamic allocas.
    // moveDynamicAllocasToUnsafeStack(F, UnsafeStackPtr, DynamicTop,
    //                                 DynamicAllocas);
    moveDynamicAllocasToHeap(F, DynamicAllocas);
    if (OptOnlyNoDynAllocas)
      return !DynamicAllocas.empty();

    // Don't treat dynamic allocas again below since they are now removed
    DynamicAllocas.clear();

    // Divide allocas into sized stacks.
    struct StackInfo {
      SmallVector<AllocaInst *, 8> StaticAllocas;
      SmallVector<Argument *, 2> ByValArguments;
    };
    StringMap<StackInfo> UnsafeStacks;
    for (AllocaInst *AI : StaticAllocas){
      UnsafeStacks[RT.getStackID(*AI)].StaticAllocas.push_back(AI);
    }
    for (Argument *Arg : ByValArguments){
      UnsafeStacks[RT.getStackID(*Arg)].ByValArguments.push_back(Arg);
    }

    // Create a base pointer for each sized stack and transform allocas.
    // IRBuilder<> IRB(F.getContext());
    Value *ClassicalUnsafeStackPtr = UnsafeStackPtr;
    for (auto &it : UnsafeStacks) {
      StringRef StackID = it.getKey();
      const struct StackInfo &Stack = it.getValue();

      // Reset IRBuilder to function entry point.
      IRB.SetInsertPoint(&F.front(), F.begin()->getFirstInsertionPt());


      Value *UnsafeStackPtr = ClassicalUnsafeStackPtr;

      // Load the current stack pointer (we'll also use it as a base pointer).
      // FIXME: use a dedicated register for it?
      Instruction *BasePointer =
          IRB.CreateLoad(StackPtrTy, UnsafeStackPtr, false, "unsafe_stack_ptr_" + StackID);
      BasePointer->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));

      assert(BasePointer->getType() == StackPtrTy);

      SmallVector<std::tuple<Value*, uint64_t, uint8_t>, 8> TaggedStackAllocas;

      // The top of the unsafe stack after all unsafe static allocas are
      // allocated.
      Value *StaticTop =
          moveStaticAllocasToUnsafeStack(IRB, F, Stack.StaticAllocas, Stack.ByValArguments,
                                        BasePointer, StackGuardSlot,
                                        UnsafeStackPtr, StackID,
                                        TaggedStackAllocas);

      // Safe stack object that stores the current unsafe stack top. It is updated
      // as unsafe dynamic (non-constant-sized) allocas are allocated and freed.
      // This is only needed if we need to restore stack pointer after longjmp
      // or exceptions, and we have dynamic allocations.
      // FIXME: a better alternative might be to store the unsafe stack pointer
      // before setjmp / invoke instructions.
      createStackRestorePoints(
          IRB, F, StackRestorePoints, StaticTop, false,
          UnsafeStackPtr, StackID);

      Type *I64Ty = Type::getInt64Ty(F.getContext());
      Type *I8Ty = Type::getInt8Ty(F.getContext());
      FunctionCallee TagMemoryFn = F.getParent()->getOrInsertFunction(
          "__subsan_tag_memory", I64Ty, I64Ty, I64Ty, I8Ty);

      // Restore the unsafe stack pointer before each return.
      for (Instruction *RI : Returns) {

        IRB.SetInsertPoint(RI);
        for (auto &T : TaggedStackAllocas) {
          Value *AddrInt = std::get<0>(T);
          uint64_t ExactSize = std::get<1>(T);
          if (ClSubsanInlineTagMemory) {
            emitInlineTagMemory(IRB, F.getParent(), AddrInt, ExactSize, 0);
          } else {
            IRB.CreateCall(TagMemoryFn,
                {AddrInt, ConstantInt::get(I64Ty, ExactSize),
                 ConstantInt::get(I8Ty, 0)});
          }
        }
        Instruction *IStore = IRB.CreateStore(BasePointer, UnsafeStackPtr);
        IStore->setMetadata(F.getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F.getContext(), std::nullopt));
      }
    }

    // even if there are no unsafe stacks in this function, make sure to still mark the restore points
    for(Instruction *ISRP : StackRestorePoints){
      if(RestorePoints.find(ISRP) == RestorePoints.end()){
        RestorePoints[ISRP] = {}; // empty usage
      }
    }

    if (UnsafeStacks.size() > NumUnsafeStacksPerFunction)
      NumUnsafeStacksPerFunction = UnsafeStacks.size();

    //errs() << "[SizedStack] sizedstack applied to " << F.getName()
    //                  << " with " << UnsafeStacks.size() << " unsafe stacks\n";
  }

  // checks & calls
  ObjectSizeOpts ObjSizeOpts;
  ObjSizeOpts.RoundToAlign = true;
  ObjectSizeOffsetVisitor ObjSizeVis(DL, &TLI, F.getContext(), ObjSizeOpts);
  ChecksOnFunc(F, ObjSizeVis);

  InstrumentCalls(F);
  
  if (ClSubsanMoveArgvToHeap && F.getName() == "main" && F.arg_size() >= 2) {
    Argument *argc = F.getArg(0);
    Argument *argv = F.getArg(1);
    if (argv->getNumUses() > 0) {
      IRBuilder<> builder(&F.front(), F.begin()->getFirstInsertionPt());
      Instruction *CallResult = builder.CreateCall(
          swiftsan_argv_fn, {argc, argv}, "argv_to_heap");
      CallResult->setMetadata(F.getParent()->getMDKindID("swiftsan"),
                              llvm::MDNode::get(F.getContext(), std::nullopt));
      argv->replaceUsesWithIf(CallResult, [CallResult](Use &U) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        return !I || I != CallResult;
      });
    }
  }

  return true;
}

static void replaceUsesWithCast(Value *What, Value *With) {
  if (isa<Constant>(What)) {
    Constant *WithC = cast<Constant>(With);
    if (With->getType() != What->getType())
      With = ConstantExpr::getBitCast(WithC, What->getType());
  }
  What->replaceAllUsesWith(With);
}


bool SizedStackRuntime::initialize() {
  Type *VoidTy = Type::getVoidTy(M.getContext());
  FunctionType *FnTy;

  if (OptStackShadowMem) {
    Type *BoolTy = Type::getInt1Ty(M.getContext());
    FnTy = FunctionType::get(VoidTy, {IntPtrTy, Int64Ty, BoolTy}, false);
    ShadowMemPoisonFunc = getOrInsertNoInstrumentFunction(M, "shadowmem_poison", FnTy);
  }

  FnTy = FunctionType::get(StackPtrTy, {IntPtrTy, IntPtrTy}, false);
  DynAllocFunc = getOrInsertNoInstrumentFunction(M, "dyn_alloc", FnTy);

  FnTy = FunctionType::get(VoidTy, {StackPtrTy}, false);
  DynFreeFunc = getOrInsertNoInstrumentFunction(M, "dyn_free", FnTy);
  DynFreeOptFunc = getOrInsertNoInstrumentFunction(M, "dyn_free_optional", FnTy);

  if (OptOnlyNoDynAllocas)
    return true;

  // Create a temporary stack pointer array as we do not know the size yet.
  stackPointerArray = createStackPtrArray(kUnsafeStackPtrVarTemp, 0);
  typeIndexNext = 0;

  return true;
}

bool SizedStackRuntime::finalize() {
  if (OptOnlyNoDynAllocas)
    return false;
#if ENABLE_RESTOREPOINT_ALL == 1
  std::set<Instruction*> iDone;
  // patch up restore points to clear _all_ unsafe stacks
  // when inserting for a function, instrument all other targets in the same function
  for(auto e : RestorePoints){
    Instruction *I = e.first;

    if(iDone.find(I) != iDone.end()){
      // already done: skip
      continue;
    }

    Function *F = I->getParent()->getParent();
    std::vector<size_t> toRestore;
    if(typeIndexNext > 0){ // there are unsafe stacks
      for(size_t sid = 0; sid < typeIndexNext; sid++){
        if(std::find(e.second.begin(), e.second.end(), sid) == e.second.end()){
          toRestore.push_back(sid);
        }
      }
    }

    IRBuilder<> IRB(F->getContext());
    // Reset IRBuilder to function entry point.
    for(size_t sid : toRestore){
      IRB.SetInsertPoint(&F->front(), F->begin()->getFirstInsertionPt());

      Value *UnsafeStackPtr = IRB.CreateInBoundsGEP(cursedType, stackPointerArray, {IRB.getInt32(0), IRB.getInt32(sid)});
      Instruction *BasePointer = IRB.CreateLoad(StackPtrTy, UnsafeStackPtr, false, "unsafe_stack_ptr_restore");
      BasePointer->setMetadata(F->getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F->getContext(), std::nullopt));

      // after the setjmp / landing pad: store the basepointer
      IRB.SetInsertPoint(I->getNextNode());
      Instruction *IStore = IRB.CreateStore(BasePointer, UnsafeStackPtr);
      IStore->setMetadata(F->getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F->getContext(), std::nullopt));

      for(auto other : RestorePoints){
        if(e.first == other.first) continue; // itself
        if(iDone.find(other.first) != iDone.end()){
          // already done: skip
          continue;
        }
        Function *FOther = other.first->getParent()->getParent();
        if(F == FOther){
          IRB.SetInsertPoint(other.first->getNextNode());
          Instruction *IStore = IRB.CreateStore(BasePointer, UnsafeStackPtr);
          IStore->setMetadata(F->getParent()->getMDKindID("swiftsan"), llvm::MDNode::get(F->getContext(), std::nullopt));
          iDone.insert(other.first);
        }
      }
    }
    iDone.insert(I);
  }
#endif

  // Replace temporary stack pointer array with a properly sized one.
  GlobalVariable *stackPointerArrayFinal = createStackPtrArray(kUnsafeStackPtrVarFinal, typeIndexNext);
  replaceUsesWithCast(stackPointerArray, stackPointerArrayFinal);
  stackPointerArray->eraseFromParent();
  stackPointerArray = nullptr;

  GlobalVariable *stackPointerArrayStatic = dyn_cast_or_null<GlobalVariable>(M.getNamedValue(kUnsafeStackPtrVar));
  if (stackPointerArrayStatic) {
    replaceUsesWithCast(stackPointerArrayStatic, stackPointerArrayFinal);
    stackPointerArrayStatic->eraseFromParent();
  }

  // Replace static library size class array with a properly sized and initialized one.
  // The array might not exist; it gets optimized out when DISABLE_SLOWPATH is
  // set in the static library
  GlobalVariable *sizeClassArrayStatic = dyn_cast_or_null<GlobalVariable>(M.getNamedValue(kUnsafeStackSizeClassesVar));
  if (sizeClassArrayStatic) {
    GlobalVariable *sizeClassArrayFinal = createSizeClassArray(kUnsafeStackSizeClassesVarFinal);
    replaceUsesWithCast(sizeClassArrayStatic, sizeClassArrayFinal);
    sizeClassArrayStatic->eraseFromParent();
  }

  // Provide stack pointer count and assigned size classes to static library.
  createStackPtrCount(kUnsafeStackPtrCountVar, typeIndexNext);




  return true;
}

size_t SizedStackRuntime::getTypeIndex(StringRef typeId) {
  auto typeIndexIt = typeIndexByTypeId.find(typeId);
  size_t typeIndex;

  // Find size class in type id.
  uint64_t SizeClass = 0;
  int idx = typeId.rfind('_');
  if (idx >= 0)
    SizeClass = std::stoll(typeId.substr(idx + 1).str());

  if (typeIndexIt == typeIndexByTypeId.end()) {
    typeIndex = typeIndexNext++;
    LLVM_DEBUG(dbgs() << "[SizedStack] getTypeIndex typeId=" << typeId <<
               " sizeclass=" << SizeClass << " typeIndex=" << typeIndex << "\n");
    typeIndexByTypeId[typeId] = typeIndex;
    assert(AssignedSizeClasses.size() == typeIndex);
    AssignedSizeClasses.push_back(SizeClass);
    ++NumUnsafeStacks;
  } else {
    typeIndex = typeIndexIt->second;
    assert(AssignedSizeClasses[typeIndex] == SizeClass);
  }

  return typeIndex;
}

__attribute__((unused))
Value *SizedStackRuntime::getOrCreateUnsafeStackPtr(IRBuilder<> &IRB, Function &F, StringRef typeId) {
  size_t typeIndex = getTypeIndex(typeId);
  return IRB.CreateInBoundsGEP(cursedType, stackPointerArray, {IRB.getInt32(0), IRB.getInt32(typeIndex)});
}

GlobalVariable *SizedStackRuntime::createStackPtrArray(StringRef varName, size_t count) {
  // We need an array of count stack pointers.
  ArrayType *type = ArrayType::get(StackPtrTy, count);
  cursedType = type;
  LLVM_DEBUG(dbgs() << "[SizedStack] create stack pointer array " << varName << " count=" << count << "\n");

  // Initialize every stack pointer to NULL.
  Constant *initElement = ConstantPointerNull::get(StackPtrTy);
  SmallVector<Constant*, 16> initElements;
  for (size_t i = 0; i < count; ++i)
    initElements.push_back(initElement);
  Constant *init = ConstantArray::get(type, initElements);

  // Create the global var.
  return new GlobalVariable(
      /*Module=*/M,
      /*Type=*/type,
      /*isConstant=*/false,
      /*Linkage=*/GlobalValue::PrivateLinkage,
      /*Initializer=*/init,
      /*Name=*/varName,
      /*InsertBefore=*/nullptr,
      /*ThreadLocalMode=*/GlobalValue::InitialExecTLSModel);
      // TODO try this, should work for LTO: /*ThreadLocalMode=*/GlobalValue::LocalExecTLSModel);
}

GlobalVariable *SizedStackRuntime::createStackPtrCount(StringRef varName, size_t count) {
  GlobalVariable *gv = dyn_cast_or_null<GlobalVariable>(M.getNamedValue(varName));
  if (!gv) return nullptr;
  IntegerType *type = IntegerType::get(M.getContext(), 64);
  Constant *init = ConstantInt::get(type, count);
  gv->setInitializer(init);
  gv->setConstant(true);
  gv->setLinkage(GlobalValue::PrivateLinkage);
  return gv;
}

GlobalVariable *SizedStackRuntime::createSizeClassArray(StringRef varName) {
  ArrayType *type = ArrayType::get(Int64Ty, AssignedSizeClasses.size());

  SmallVector<Constant*, 16> initElements;
  for (uint64_t SC : AssignedSizeClasses)
    initElements.push_back(ConstantInt::get(Int64Ty, SC));
  Constant *init = ConstantArray::get(type, initElements);

  return new GlobalVariable(
      /*Module=*/M,
      /*Type=*/type,
      /*isConstant=*/true,
      /*Linkage=*/GlobalValue::PrivateLinkage,
      /*Initializer=*/init,
      /*Name=*/varName,
      /*InsertBefore=*/nullptr,
      /*ThreadLocalMode=*/GlobalValue::InitialExecTLSModel);
}

uint64_t SizedStackRuntime::IncludeRedzoneSize(uint64_t Size) {
  uint64_t Res = 0;
  if (Size <= 4)  Res = 16;
  else if (Size <= 16) Res = 32;
  else if (Size <= 128) Res = Size + 32;
  else if (Size <= 512) Res = Size + 64;
  else if (Size <= 4096) Res = Size + 128;
  else                   Res = Size + 256;
  return Res;
}

uint64_t SizedStackRuntime::roundUpToSizeClass(uint64_t Size) {
  // ASan stack redzone sizes
  uint64_t Rounded = 0;
  for (const uint64_t SC : kSizeClasses) {
    if (Size <= SC) {
      Rounded = SC;
      break;
    }
  }
  if (Rounded == 0) {
    assert(IncludeRedzoneSize(Size) <= kDefaultUnsafeStackSize);
    // for (Rounded = Size; Rounded % StackAlignment; ++Rounded);
    Rounded = round_to_power_of_two(Size);
    assert(Rounded % StackAlignment == 0);
    LLVM_DEBUG(dbgs() << "[SizedStack] creating new size class " << Rounded
                      << " for " << Size << "-byte allocation\n");
  }
  return Rounded;
}

std::string SizedStackRuntime::getStaticStackID(uint64_t Size, const Value &V) {
  uint64_t PaddedSize = roundUpToSizeClass(IncludeRedzoneSize(Size));
  assert(PaddedSize > 0);
  std::string id;
  raw_string_ostream ss(id);
  ss << "static_" << PaddedSize;
  return id;
}

uint64_t SizedStackRuntime::getStaticAllocaAllocationSize(const AllocaInst* AI) {
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

std::string SizedStackRuntime::getStackID(const AllocaInst &AI) {
  assert(AI.isStaticAlloca());
  return getStaticStackID(getStaticAllocaAllocationSize(&AI), AI);
}

std::string SizedStackRuntime::getStackID(const Argument &Arg) {
  // Type *Ty = Arg.getType()->getPointerElementType();
  Type *Ty = Arg.getParamByValType();
  return getStaticStackID(DL.getTypeStoreSize(Ty), Arg);
}

class SafeStackLegacyPass : public ModulePass {
  const TargetMachine *TM = nullptr;

public:
  static char ID; // Pass identification, replacement for typeid..

  SafeStackLegacyPass() : ModulePass(ID) {
    initializeSafeStackLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    // range opt
    AU.addRequired<TargetTransformInfoWrapperPass>();
    // AU.addRequired<DependenceAnalysisWrapperPass>();
  }

  bool runOnFunction(Function &F, SizedStackRuntime &RT) {
    LLVM_DEBUG(dbgs() << "[SafeStack] Function: " << F.getName() << "\n");

    if (!F.hasFnAttribute(Attribute::SafeStack)) {
      LLVM_DEBUG(dbgs() << "[SafeStack]     safestack is not requested"
                           " for this function\n");
      return false;
    }

    if (F.isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[SafeStack]     function definition"
                           " is not available\n");
      return false;
    }
    if (F.hasFnAttribute(llvm::Attribute::DisableSanitizerInstrumentation)){
        errs() << "Not instrumenting function (Disable San): " << F.getName() << "\n";
        return false;
    }

    TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    auto *TL = TM->getSubtargetImpl(F)->getTargetLowering();
    if (!TL)
      report_fatal_error("TargetLowering instance is required");

    auto *DL = &F.getParent()->getDataLayout();
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto &ACT = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

    // Compute DT and LI only for functions that have the attribute.
    // This is only useful because the legacy pass manager doesn't let us
    // compute analyzes lazily.

    DominatorTree *DT;
    // bool ShouldPreserveDominatorTree;
    std::optional<DominatorTree> LazilyComputedDomTree;

    // // Do we already have a DominatorTree avaliable from the previous pass?
    // // Note that we should *NOT* require it, to avoid the case where we end up
    // // not needing it, but the legacy PM would have computed it for us anyways.
    // if (auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>()) {
    //   DT = &DTWP->getDomTree();
    //   // ShouldPreserveDominatorTree = true;
    // } else {
    //   // Otherwise, we need to compute it.
    //   LazilyComputedDomTree.emplace(F);
    //   DT = &*LazilyComputedDomTree;
    //   // ShouldPreserveDominatorTree = false;
    // }

    DT = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

    LoopInfo LI(*DT);

    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);

    ScalarEvolution SE(F, TLI, ACT, *DT, LI);

    DominanceFrontier DF;
    DF.analyze(*DT);

    // DependenceInfo *DI = &getAnalysis<DependenceAnalysisWrapperPass>(F).getDI();

    AliasAnalysis *AA = &getAnalysis<AAResultsWrapperPass>(F).getAAResults();

    bool r = SafeStack(F, *TL, *DL, &DTU, SE, *DT, DF, RT, TLI, LI, AA/*, DI*/).run();

    const TargetTransformInfo *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

    if(!F.hasOptNone()){
      // Recalculate DTU for simplifyCFG 
      DT->recalculate(F);
      DTU.recalculate(F);

      for (BasicBlock &BB : F) {
        // Cleanup possible branch to unconditional branch (from conditional range optimizations)
        simplifyCFG(&BB, *TTI, &DTU);
      }
    }
    return r;
  }


  // Get current date/time, format is YYYY-MM-DD.HH:mm:ss
  const std::string currentDateTime() {
      time_t     now = time(0);
      struct tm  tstruct;
      char       buf[80];
      tstruct = *localtime(&now);
      // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
      // for more information about date/time format
      strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

      return buf;
  }

  bool runOnModule(Module &M) override {

    if(ClDelayInstrumentation){
      errs() << "ClDelayInstrumentation was set, two-stage LTO... do not instrument yet\n";
      return false;
    }

#ifndef USE_GOLD_PASSES
    if (!ClSizedStack) return false;
#endif
    Triple TargetTriple = Triple(M.getTargetTriple());
    if(TargetTriple.getArch() == Triple::x86_64){
      // [subsan x86_64 LAM port 2026-05-14] explicit pointer tagging via Intel
      // LAM_U57: 6-bit tag in bits 62:57, bit 63 reserved. Runtime enables LAM
      // mode via arch_prctl(ARCH_ENABLE_TAGGED_ADDR, 6).
      isImplicitTagging = false;
      isX86 = true;
      subsanTagShift       = 57;
      subsanFastPathMask   = 0xFE00000000000000ULL;  // bits 63:57
      subsanGranuleIdxMask = 0x001FFFFFFFFFFFFFULL;  // (1<<53)-1
      // 6-bit tag (0=untagged, 1..63 = 63 entries) split into three namespaces:
      subsanStackTagLo  = 1;   subsanStackTagHi  = 21;
      subsanGlobalTagLo = 22;  subsanGlobalTagHi = 42;
      // heap range (43..63) lives in the compiler-rt TLS counter.
    }
    else if(TargetTriple.isAArch64()){
      isImplicitTagging = false;
      isX86 = false;
      subsanTagShift       = 56;
      subsanFastPathMask   = 0xFFFF000000000000ULL;
      subsanGranuleIdxMask = 0x000FFFFFFFFFFFFFULL;
      subsanStackTagLo  = 1;    subsanStackTagHi  = 85;
      subsanGlobalTagLo = 86;   subsanGlobalTagHi = 170;
    }
    else{
      errs() << "SafeStack: Unsupported Arch\n";
      return false;
    }

    if (TargetTriple.isAArch64()) {
      for (GlobalVariable &GV : M.globals()) {
        if (GV.isThreadLocal()) {
          GV.setThreadLocalMode(GlobalVariable::LocalExecTLSModel);
        }
      }
    }

    bool isSafeStack = false;
    for (Function &F : M) {
      if (F.hasFnAttribute(Attribute::SafeStack)) {
          isSafeStack = true;
      }
    }

    if(!isSafeStack){
      return false;
    }

#define DUMP_IR 0
#if DUMP_IR == 1
    std::string modname(M.getName());
    std::error_code ec;
    llvm::raw_fd_ostream OS(modname + "-pre.txt", ec);
    M.print(OS, nullptr);
#endif

    bool Changed = false;

    SizedStackRuntime RT(M);
    Changed = RT.initialize();

    for (Function &F : M) {
      bool instrument = shouldInstrument(F);
      if (instrument) {
        Changed |= runOnFunction(F, RT);
      }
    }

#if 0
    FILE* fp = fopen("/tmp/safe_stack_sizes.txt", "a");
    std::string CurrWorkPath = std::getenv("PWD");
    std::string FullFilename = CurrWorkPath + "/" + M.getSourceFileName();
    fprintf(fp, "%s %lu %lu\n", FullFilename.c_str(), NumUnsafeStacks.getValue(), NumUnsafeStacksPerFunction.getValue());
    //errs() << "[SizedSafeStack analysis]: " << FullFilename << "\n" << NumUnsafeStacks << " " << NumUnsafeStacksPerFunction << "\n";
    fflush(fp);
    fclose(fp);
#endif

    Changed |= RT.finalize();

    GlobalRedzones GR;
    GR.runOnModule(M);

#if DUMP_IR == 1
    llvm::raw_fd_ostream OS2(/*currentDateTime() +*/ modname + "-post.txt", ec);
    M.print(OS2, nullptr);
#endif

    return Changed;
  }
};

} // end anonymous namespace

char SafeStackLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(SafeStackLegacyPass, DEBUG_TYPE,
                      "Safe Stack instrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
// INITIALIZE_PASS_DEPENDENCY(DependenceAnalysisWrapperPass)
INITIALIZE_PASS_END(SafeStackLegacyPass, DEBUG_TYPE,
                    "Safe Stack instrumentation pass", false, false)

ModulePass *llvm::createSafeStackPass() { return new SafeStackLegacyPass(); }
