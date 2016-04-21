# Introduction #

In order to process images on the graphics card (GPU) the input image must first be in the device memory. This is because the GPU cannot access the host memory. Unfortunately copying memory from the host to the device is a very slow process. In a simple kernel it often takes longer to copy the memory across then it does to operate on it. The goal of the CITK modified ITK architecture was to provide the input image on the device, whilst performing minimal host to device memory transfers.

# Possible Solutions #

In the past, many other groups have attempted to create a GPU enabled ITK. At the commencement of this project none had achieved this. Here are some of the approaches they took, their successes and their shortcomings.

### Attempt 1: Branch the Pipeline ###

One such solution was to break the pipeline at the beginning of the filter execution. Once the filter was executed, via an update method, the filter would copy the image across to the device. The filter would then pass a pointer to the device memory to a kernel, which was then executed. Following this the processed image would then be copied back from the device to the host and inserted back into the pipeline.

This solution copied the image to the device and back to host on a per filter basis. The problem with this is you are guaranteed a host to device and a device to host memory transfer on every filter. In a situation where multiple filters are used, or a filter is used multiple times, this is not acceptable. For instance, Registration can call a metric 400 times, this would result in 800 memory transfers.

This solution also leaves the full responsibility of memory transfers up to the developers. This results in redundant code between filters and requires any developer to have a thorough understanding of both the ITK architecture and the GPU architecture.

### Attempt 2: Interfaces ###

This solution overcame the excess memory transfers found in branching the pipeline by using an interface. This interface was placed in the pipeline by the end-user before using a GPU based filter, and another interface was placed after. The first interface copied the image to the device and the second interface copied it back.

This meant multiple GPU filters could be ran in succession without multiple memory transfers. No matter how many filters were used, you were always guaranteed the minimum number of memory transfers. The downside to this approach was it was up to the end user to implement these interfaces. A user needed to be aware whether the filters were GPU enabled or not, and which interfaces were used where.

This solution also suffered in a situation where GPU and CPU filters were used in combination as several interfaces were needed. This complicated the user experience and left most novice users in the dark.

# The CITK Solution #

The CITK solution was to modify the underlying ITK architecture. The fundamental component of the pipeline is the ITK Image class. Within this class is a pixel container called ImportImageContainer, used to manage the image data. CITK includes a substitute pixel container named CudaImportImageContainer. This pixel container has all the same functionality of the ImportImageContainer which results in full compatibility with existing ITK components.

The CudaImportImageContainer not only managed the image data on the host, but it also managed the image data on the device. When a standard filter requested the image data, such as through an iterator, the CudaImportImageContainer checks whether the most up to date image is on the device or the host. If it is on the device, it is copied back onto the host. This data is then supplied to the user. Similarly when a GPU filter requests the image data, the CudaImportImageContainer would check where the most up to date image is, and copy it to the device if required.

The CudaImportImageContainer can track where the most up to the date image is by which set command was used last, and assumes when a standard iterator requests the data that is modifies it.

The result of this is memory transfers are only performed when required and are completed transparent to both the developer and the user. This leaves all the responsibility on the architect, rather than the developer or the user such as in the other attempts.

# Usage #

### Developers ###

In order to use the modified architecture, a developer need only use Get and Set methods. These methods alter a pointer to global device memory and keep track of where the most up to date image is. This means a developer only needs the standard knowledge ITK expects, and a knowledge of CUDA in order to write a filter for the CITK.

### Users ###

Users can use CITK as they would standard ITK. The only requirement is to extract CITK into the ITK folder and build ITK as normal. As long as they have a CUDA enabled graphics card they can begin using the CITK filters immediately. The API of the CITK filters is designed to be identical to their standard ITK counterparts, meaning existing ITK applications need only have their class names changed.