# Introduction #

These are the basic commands supplied by the modified architecture.

# Functions #

**Get global device pointer of input image.**

```
input->GetDevicePointer();
```

**Set global device pointer of output image.**

```
output->GetPixelContainer()->SetDevicePointer(ptr, N, true);
```

**Don't free global device pointer of input image after use.**

```
inputPtr->GetPixelContainer()->SetContainerManageDevice(false);
```