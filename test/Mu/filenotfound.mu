// RUN: not muc --emit=mlir nofilebythisname.mu 2>&1 | FileCheck %s
// RUN: not muc --emit=mlir nofilebythisname 2>&1 | FileCheck %s --check-prefixes=CHECK-NOMU

// CHECK: File not found: nofilebythisname.mu
// CHECK-NOMU: Invalid filetype: nofilebythisname
// CHECK-NOMU-NEXT: Files given to muc must end in .mu
