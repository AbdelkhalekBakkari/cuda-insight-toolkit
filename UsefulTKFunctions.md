# Introduction #

Working with CUDA requires low level data structures. To find pointers to these values often required a developer to chase the ITK hierarchy. To make life a little easier, here is a list of functions to retrieve the low level data structures of some commonly used values.

# Useful Functions #

| Name | Function |
|:-----|:---------|
| Structuring Element Kernel | m\_Kernel.GetBufferReference().begin(); |
| Image Dimensions | input->GetLargestPossibleRegion().GetSize().GetSize(); |
| Radius Dimensions | m\_Radius.GetSize(); |
| Number of Pixels | input->GetPixelContainer()->Size(); |
| Number of Dimensions | input->GetLargestPossibleRegion().GetImageDimension(); |