# Cuda Insight Toolkit (CITK) #

CITK is an extension to the ITK architecture to support General-purpose computing on graphics processing units. The chosen medium for this extension is the Compute Unified Device Architecture (CUDA). The advantages of using graphic cards for image processing are well documented and CITK has been no exception to these findings.

### Insight Segmentation and Registration Toolkit (ITK) ###

<p align='center'>
"ITK is an open-source, cross-platform system that provides developers with an extensive suite of software tools for image analysis.  Developed through extreme programming methodologies, ITK employs leading-edge algorithms for registering and segmenting multidimensional data. The goals for ITK include:<br>
<br>
<ul><li>Supporting the Visible Human Project.<br>
</li><li>Establishing a foundation for future research.<br>
</li><li>Creating a repository of fundamental algorithms.<br>
</li><li>Developing a platform for advanced product development.<br>
</li><li>Support commercial application of the technology.<br>
</li><li>Create conventions for future work.<br>
</li><li>Grow a self-sustaining community of software users and developers."</li></ul>

<blockquote><i>Taken from <a href='http://itk.org'>ITK</a></i> </p></blockquote>

### The Project ###

CITK aims to assist ITK in achieving these goals by extending their architecture to support General-purpose computing on graphics processing units. CITK also provides many filters which utilize this extended architecture and are currently developing a Registration framework based on it.

The results from the CITK filter development show anywhere from a 5x speedup (from AddImageFilter) to a 800x speedup (from MeanImageFilter). The performance of CITK varies slightly based on the hardware and drastically based on its application. This is better detailed in [CITK Performance Tips](CITKPerformanceTips.md).

The CITK project is completely open source and welcomes anyone interested in helping develop it. Mailing lists are available for both users and developers of the software. Some Australian Universities have also shown an interest in using this architecture to further develop ITK in new and exciting directions such as Scale-invariant feature transform (SIFT).

Although in its early stages, CITK shows great potential as an easily adaptable solution to enabled ITK to use GPGPU technology. It provides an easy to use and intuitive interface for the developers whilst being completely transparent to the end user. Existing users of ITK need only acquire a CUDA enabled graphics card, extract the code and rebuild ITK in order to see the benefits of this technology.

### Origins ###

CITK was originally started as a summer student internship program based at Victorian Partnership for Advanced Computing [VPAC](http://www.vpac.org). The program was supervised by Mike Kuiper of VPAC and directed by Richard Beare of Monash University. The students developing the software were Daniel Micevski, Chris Share, Luke Parkinson and Phillip Ward.

At the conclusion of the project the developers got in contact with ITK and an interest was shown. [VPAC](http://www.vpac.org) decided to continue to support the developers after the program concluded and has been behind CITK ever since.

### Development ###

CITK follows the ITK style requirements and all CITK class names mimic their ITK counterparts with a CUDA prefix. It maintains all existing ITK features and provides the developer with access to the input image already on the device. This input image is referenced by a pointer and can be used within a CUDA kernel to execute the required algorithm. CITK also allows the developer to pass a pointer to the processed image on either device memory or main memory. This is then passed down the pipeline in the same fashion as the standard ITK architecture.
