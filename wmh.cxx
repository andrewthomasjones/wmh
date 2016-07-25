/* W2MHS in ITK
 *
 * 
 * 
 * 
 * additional edits by andrew jones
 * andrewthomasjones@gmail.com
 * 2016
 * main changes:
 * input arg structure changed
 * new modes for testig and training
 * new models
 * added boost
 * refactored to use strings over *char mostly
 */
 
 // based on
 
 /*
 *  re-implementation of W2MHS + improvements in ITK
 * 
 *  Shahrzad Moeiniyan Bagheri - shahrzad.moeiniyanbagheri@uqconnect.edu.au
 *  Andrew Janke - a.janke@gmail.com
 *  Center for Advanced Imaging
 *  The University of Queensland
 *
 *  Copyright (C) 2015 Shahrzad Moeiniyan Bagheri and Andrew Janke
 *  This package is licenced under the AFL-3.0 as per the original 
 *  W2MHS MATLAB implmentation.
 */
 
//ITK HEADERS
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkNiftiImageIO.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhood.h"
#include "itkConvolutionImageFilter.h"
#include "itkConstantBoundaryCondition.h"
#include "itkSubtractImageFilter.h"
#include "itkNeighborhoodOperator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkHistogramThresholdImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkInvertIntensityImageFilter.h"
#include "itkBinaryContourImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkPowImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkNeighborhoodOperator.h"

//MISC
#include "array"
#include "iostream"
#include "fstream"
#include "cstdio"
#include "cstdlib"
#include "cv.h"       // opencv general include file
#include "ml.h"
#include "string.h"
#include "vector"
#include "unistd.h"
#include "getopt.h"
#include "vector"


//BOOST
#include "boost/math/distributions/students_t.hpp"
#include "boost/program_options.hpp"
#include "boost/math/tools/roots.hpp"
#include "boost/math/special_functions/digamma.hpp"

//NAMESPACES
using namespace cv;
using namespace boost::program_options;
using namespace boost::math;

//IMAGE TYPES
typedef itk::Image<float,3> ImageType;
typedef ImageType::Pointer ImagePointer;
typedef itk::Image<float,2> ImageType2D;
typedef ImageType2D::Pointer ImagePointer2D;
typedef std::pair <std::vector<double>, std::vector<double> > vecPair; 

//FILTER, ETC TYPES
typedef itk::MaskImageFilter< ImageType, ImageType > MaskFilterType;
typedef itk::ConstantBoundaryCondition<ImageType>  BoundaryConditionType;
typedef itk::ConstNeighborhoodIterator< ImageType , BoundaryConditionType> NeighborhoodIteratorType;
typedef itk::Neighborhood< ImageType > NeighborhoodType;
typedef itk::ConvolutionImageFilter<ImageType> ConvolutionType;

//declare the methods' signature here
ImagePointer NiftiReader(char * inputFile);
ImagePointer NiftiReader(std::string inputFile);
bool NiftiWriter(ImagePointer input,std::string outputFile);


//legacy functions
void QuantifyWMHs(float pmapCut, ImagePointer pmapImg, std::string ventricleFilename, std::string outputFilename);
ImagePointer MultiplyTwoImages(ImagePointer input1,ImagePointer input2);
void MarginateImage(ImagePointer input,int marginWidth);
ImagePointer InvertImage(ImagePointer input, int maximum);
ImagePointer PowerImageToConst(ImagePointer input, float constPower);
ImagePointer ThresholdImage(ImagePointer input, float lowerThreshold, float upperThreshold);
float SumImageVoxels(ImagePointer input);


////general methods
void ShowCommandLineHelp();


//New functions - Andy 2016
std::string renamer(std::string baseName, std::string suffix);
vecPair fitTdist(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter, double a, double b);
vecPair fitMest(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter);
ImagePointer ClassifyWMHsT(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour,  double p_thresh_const, double min_df, double max_df);
ImagePointer ClassifyWMHsM(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter, double tol);


//math functions
double phi(double d);
template <typename T> int sgn(T val);
double df_eq_est (double v, double v_sum);

//function class with operator for root finder for df of t-dist
class df_eq_func{
	public:
		df_eq_func(double v) :  v_sum(v){};
	
		double getSumV();
		
		double operator()(double v) {
			
			double temp = -digamma(0.5*v) + std::log(0.5*v) + v_sum + digamma((v+1.)/2.) - std::log((v+1.)/2.)+1.;
			//std::cout << "v: " << v << " v_sum: " << v_sum << " temp: " << temp << std::endl;
			return temp;
			//return df_eq_est(v,v_sum);
			
		}
		
	private:
		double v_sum;
};


double df_eq_func::getSumV()
{
    return v_sum;
}


std::string vecToString(vector<string> v);



		
int main(int argc, char *argv[])

{		//flags
		bool TFlag = false;
		bool MFlag = false;
		bool twoDflag =  false; 
		bool compare =  false;   	   		
		
		int min_neighbours;
		int verb_lvl = 0;
		
		double p_thresh_const;
		double d_thresh_const;
		
		double a;
	    double b;
	    
	    double tol;
			
		//names of various files, input and output
		std::string WMStrippedFilename;
		std::string GMCSFStippedFilename;
		std::string VentricleBinFilename; 
		std::string BRAVOFilename;
		std::string WMMaskFilename;
				
		//output
		std::string quantResultFilename;
		std::string segOutFilename;
		
		
		try
		{
			options_description desc{"Options"};
			desc.add_options()
				("help,h", "Help screen")
				
				("min_n,n", value<int>(), "Minimum neighbours required for WMH pixels in new model ")
				("p_cut,c", value<double>(), "pmap cut-off for EV calculations in RF model")
				("p_thresh,x", value<double>(), "Significance cut off for t-dist based model. Lower values mean less pixels classified as WMH ")
				("d_thresh,d", value<double>(), "Distance cut off for robust normal (M-Estimator) model. Higher values mean less pixels classified as WMH")
				
				("min_v,a", value<double>()->default_value(0.01), "min df fotr t-dist")
				("max_v,b", value<double>()->default_value(10.), "max df fotr t-dist")
				
				("tol,t", value<double>()->default_value(0.001), "tolerance value for models")
				
				("wm_strip,w", value< vector<string>> () , "WM Stripped  filename")
				("gm_strip,g", value< vector<string> >() , "GMCSF  filename")
				("vent_mask,v", value< vector<string> >() , "ventricle binary mask filename")
								
				("quant,q", value< vector<string> >() , "Quant result filename")														
				
				("bravo,y", value< vector<string>> () , "Filename for 2nd channel image")
				("wm_mask,f", value< vector<string>> () , "Filename for WM mask to extract correct region on 2nd channel image")
				("verb,z", value<int>() , "verbosity level: 0 -> none, 1 -> brief, 2 -> all");
	
			variables_map vm;
			store(parse_command_line(argc, argv, desc), vm);
			notify(vm);
		
			if (vm.count("help")) {
				ShowCommandLineHelp();
				return 1;
			}
			
			if(vm.count("verb")){
					verb_lvl = vm["verb"].as<int>();
				}
				
			
			try{
				try{
				
				//names of various files, input and output
				WMStrippedFilename = vecToString(vm["wm_strip"].as< vector<string> >());  
				GMCSFStippedFilename = vecToString(vm["gm_strip"].as< vector<string> >()); 
				VentricleBinFilename = vecToString(vm["vent_mask"].as< vector<string> >());
				
				//load images here to errors get caught
				ImagePointer WMStrippedImage = NiftiReader(WMStrippedFilename,verb_lvl); 
				ImagePointer GMCSFStippedImage = NiftiReader(GMCSFStippedFilename,verb_lvl); 
				ImagePointer VentricleBinImage = NiftiReader(VentricleBinFilename,verb_lvl); 
				 
				min_neighbours = vm["min_n"].as<int>();  //inputs for models

				//std::cout<< p_thresh_const <<  vm["p_thresh"].as<double>() << std::endl;
				//output
				quantResultFilename = vecToString(vm["quant"].as< vector<string> >()); 
				
				}catch(...){
					std::cerr << "Mandatory arguments are missing. Run --help" <<std::endl;
					return 0; 
				}
				
				
				//two d
				if (vm.count("bravo")){
					BRAVOFilename = vecToString(vm["bravo"].as< vector<string> >()); 
					WMMaskFilename = vecToString(vm["wm_mask"].as< vector<string> >());
					ImagePointer WMMaskImage = NiftiReader(WMMaskFilename,verb_lvl); 
				    ImagePointer BRAVOImage = NiftiReader(BRAVOFilename,verb_lvl); 
					
					twoDflag = true;
				}
				
				if(vm.count("d_thresh")){
					MFlag= true;
					d_thresh_const = vm["d_thresh"].as<double>();
									
					a = vm["min_v"].as<double>();
					b = vm["max_v"].as<double>();
					
				}
				
				if(vm.count("p_thresh")){
					TFlag= true;
					p_thresh_const = vm["p_thresh"].as<double>();
				}
				
				
				
				if(vm.count("tol")){
					tol = vm["tol"].as<double>();
				}
					
			}catch(...){
				std::cerr << "Unable to read all inputs. Run --help" <<std::endl;
				return 0;
			} 
			

			
			if(verb_lvl >= 1){
				std::cout << "Inputs read successfully." <<std::endl<<std::endl;
			}
			
			if(verb_lvl == 2){
			
				for (const auto& it : vm) {
				  std::cout << it.first.c_str() << ": " ;
				  auto& value = it.second.value();
				  if (auto v = boost::any_cast<double>(&value))
					std::cout << *v;
				else if (auto v = boost::any_cast<int>(&value))
					std::cout << *v;
				  else if (auto v = boost::any_cast< vector<string> >(&value))
					std::cout << vecToString(*v);
				  else
					std::cout << "error";
				 std::cout << std::endl;
				}
			
			
			}
			
			std::cout << std::endl;
		
		}
		catch(...)
		{//fix this up later with more specific errors, currently only needs the correct args to be present for each mode.
			//doesnt check output locations have write access, if files are legit, if numbers are in range, etc etc
			std::cerr << "Unknown failure reading in cmd line args" << std::endl;
			return 0;
		}
		
		ImagePointer OutImage;
		
		
		
		//~ 
		//~ if(MFlag){
		//~ 
			//~ 
			//~ 
			//~ 
			//~ std::string quantResultFilenameM = renamer(quantResultFilename, "_m");	
			//~ std::string segOutFilenameM = renamer(segOutFilename, "_m");	
			//~ 
			//~ std::cout<< "M-estimator model"  << std::endl;
			//~ std::cout<< "quant :  " << quantResultFilenameM  << std::endl;
			//~ std::cout<< "seg :  " << segOutFilenameM  << std::endl;
											//~ 
			//~ std::cout<< "d_thresh_const :  " << d_thresh_const  << std::endl;
			//~ std::cout<< "min_neighbour :  " << min_neighbours  << std::endl;
				//~ 
			//~ //RFSegOutImage=ClassifyWMHsT(inNifti, segOutFilename,  min_neighbours,  p_thresh_const, a, b);
			//~ RFSegOutImageM=ClassifyWMHsM(inNifti, segOutFilenameM,  min_neighbours,  d_thresh_const, 100, 0.0001);
			//~ 
			//~ QuantifyWMHs(0.0, RFSegOutImageM, ventricleBinFilename, quantResultFilenameM);
			//~ //QuantifyWMHs(0.0, RFSegOutImage2, ventricleBinFilename, quantResultFilename);
			//~ 
			//~ std::cout << "Done" <<std::endl;
	//~ 
		//~ 
		//~ }
			//~ 
		//~ if(TFlag){
			//~ 
			//~ ImagePointer RFSegOutImageT; 
			//~ ImagePointer inNifti = NiftiReader(WMStrippedFilename); 
			//~ 
			//~ std::string quantResultFilenameT = renamer(quantResultFilename, "_t");	
			//~ std::string segOutFilenameT = renamer(segOutFilename, "_t");
				//~ 
			//~ std::cout<< "t-distribution model"  << std::endl;
			//~ 
			//~ std::cout<< "quant :  " << quantResultFilenameT  << std::endl;
			//~ std::cout<< "seg :  " << segOutFilenameT  << std::endl;
			//~ 
			//~ std::cout<< "p_thresh_const :  " << p_thresh_const  << std::endl;
			//~ std::cout<< "min_neighbour :  " << min_neighbours  << std::endl;
    			//~ 
			//~ RFSegOutImageT=ClassifyWMHsT(inNifti, segOutFilenameT,  min_neighbours,  p_thresh_const, a, b);
			//~ //RFSegOutImage2=ClassifyWMHsM(inNifti, segOutFilenameM,  min_neighbours,  d_thresh_const);
			//~ 
			//~ //QuantifyWMHs(0.0, RFSegOutImageM, ventricleBinFilename, quantResultFilenameM);
			//~ QuantifyWMHs(0.0, RFSegOutImageT, ventricleBinFilename, quantResultFilenameT);
			//~ 
			//~ std::cout << "Done" <<std::endl;
		//~ 
			//~ 
		//~ }
	//~ 
	
	
	return 0;
	
		
}




void ShowCommandLineHelp()
{
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-h, --help:  Shows command line help." << std::endl;
	std::cerr << "      " << "-z, --verb:  Verbosity level.  0 -> none, 1 -> brief, 2 -> all";
	std::cerr << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << "Required arguements:  " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-w, --wm_strip: Specifies the .nii file name and path of the WM stripped image."<< std::endl;
	std::cerr << "      " << "-g, --gm_strip: Specifies the .nii file name and path of the GMCSF stripped image." << std::endl;
	std::cerr << "      " << "-v, --vent_mask: Specifies the .nii file name and path of the binary ventricle PVE image." << std::endl;
	std::cerr << "      " << "-q, --quant: Specifies the (.txt) file name and path, where WMH quantification results will be saved." << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " <<  "  *** Must supply at least one of -x, -d as these also function to select the relevant model. ***" << std::endl;
	std::cerr << "=============================================================================="<< std::endl; 
	std::cerr << "Model control: " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-n, --min_n: Specifies the minimum neighbours for new methods"<< std::endl;
	std::cerr << "      " << "-x, --p_thresh:  Significance cut off for t-dist based model." << std::endl;
	std::cerr << "                      " << "Lower values mean fewer pixels classified as WMH" << std::endl;
	std::cerr << "      " << "-d, --d_thresh:  Distance cut off for robust normal (M-Estimator) model." << std::endl;
	std::cerr << "                      " << "Higher values mean fewer pixels classified as WMH" << std::endl;
	std::cerr << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << "If using a second channel image:  " << std::endl;
	std::cerr << std::endl;
	std::cerr << "      " << "-y, --bravo:  Filename for 2nd channel image." << std::endl;
	std::cerr << "      " << "-f, --wm_mask:  Filename for WM mask to extract correct region on 2nd channel image" << std::endl;
	std::cerr << "=============================================================================="<< std::endl;
	std::cerr << "Algorithm control: " << std::endl;
	std::cerr  << std::endl;
	std::cerr << "      " << "-a, --min_v:  minimum degrees of freedom for t-distribution fit. Default: 0.01" << std::endl;
	std::cerr << "      " << "-b, --max_v:  maximum degrees of freedom for t-distribution fit. Default: 10" << std::endl;
	std::cerr << "      " << "-t, --tol:  Convergence tolerance. Default: 0.001." << std::endl;
	std::cerr  << std::endl;
				
				
				
	


}//end of ShowCommandLineHelp()




double phi(double d){
	double c = 1.345;
	double temp = 0.0;
		
		
	if(abs(d) < c) {
		temp = d;
	}else{
		temp= c*sgn(d);
	}
		
		
	return temp/d; 
}


 template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
}

//~ double df_eq_est(double v, double v_sum){
	 //~ return -Digamma(0.5*v) + std::log(0.5*v) + v_sum + Digamma((v+1.)/2.) - std::log((v+1.)/2.)+1.;
//~ }

std::string	vecToString(vector<string> v){
		std::string s;
		for (const auto &piece : v) s += piece;
		return s;
	}




//FOR TEST: 'testingSamples' IS PASSED TO THIS METHOD ONLY WHEN TESTING & DEBUGGING. IN REAL CASES, THESE SAMPLES WILL BE EXTRACTED FROM THE INPUT IMAGES.
//ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,char *rfSegOutFilename, Mat testingSamples)
ImagePointer ClassifyWMHsT(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double p_thresh_const, double a, double b)
{
//alg parameters
	int max_iter=100;
	double tol = 0.0001;
	double v_init =4;
		

   std::cout << "Performing WMH segmentation t-dist ..." << std::endl;

   MarginateImage(WMModStripImg,5);                        
   
   //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.

   /* The image will be rescaled in order to discard the long tail seen in the image histogram (i.e. about 0.3 of the histogram, which contains informationless voxels).
    * Note: itk::BinaryThresholdImageFilter can be used instead, if it is required to save the thresholded indexes (as in the W2MHS toolbox).
    */
   
   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   //itk::ImageRegionIterator<ImageType> inputIterator(WMModStripImg, WMModStripImg->GetRequestedRegion());

   //creating an output image (i.e. a segmentation image) out of the prediction results.
   ImagePointer RFSegOutImage=ImageType::New();

   ImageType::IndexType outStartIdx;
   outStartIdx.Fill(0);

   ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
   ImageType::RegionType outRegion;
   outRegion.SetSize(outSize);
   outRegion.SetIndex(outStartIdx);

   RFSegOutImage->SetRegions(outRegion);
   RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
   RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
   RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
   RFSegOutImage->Allocate();
  
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
     
   NeighborhoodIteratorType::RadiusType radius;
   radius.Fill(1);
    
   // set up iterators
    itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
 
	
    NeighborhoodIteratorType inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
  	
  	std::vector<double> y;
  	std::vector<double> zeros;
  	//model fit
  	// robust fit normal dist with M estimator
  	
	while(!inputIterator.IsAtEnd())
	{
		y.push_back(inputIterator.GetCenterPixel());
		if(inputIterator.GetCenterPixel()>0.0){
			zeros.push_back(1.0);
		}else{
			zeros.push_back(0.0);
		}
		++inputIterator;
	}
	
	
	vecPair model = fitTdist( y,  zeros,   tol,  max_iter,  a,  b);
	
	
	double mu = model.second.at(2);
	double sigma = model.second.at(1);   
	double v = model.second.at(0);
	
	//std::cout << "df  " << v << " mean "<< mu << " sd  "<< sigma << std::endl;
	
	double p_thresh_const_2 = 1.0-p_thresh_const;
		
	// Construct a students_t distribution with v degrees of freedom:
    students_t d1(v);
    std::vector<double> q(y.size(), 0.0);
    std::transform(y.begin(), y.end(), q.begin(),  [d1, mu, sigma](double x) { return cdf(d1, (x-mu)/std::sqrt(sigma)); }); 
	
  	vector<double>::iterator it; 
    it=q.begin();
	   
	inputIterator.GoToBegin();
	
	int c = (inputIterator.Size());
	int mid = (inputIterator.Size()) / 2;
	
	
	
	while(!inputIterator.IsAtEnd())
	{
		double acc = 0.0;
		double neighbourhood_mean = 0.0;
		int non_zero = 0;
		
		for(int i=0; i<c; i++){		
			double temp  = inputIterator.GetPixel(i);
			if(i != (c/2))
			{
				acc += temp;
				if(temp>0){++non_zero;}
			}
		
		}	
		neighbourhood_mean = acc/non_zero;			       
        
         if(neighbourhood_mean > mu & *it > (1.- p_thresh_const)){
			RFSegOutIterator.Set(1.0);
		 }else{
			RFSegOutIterator.Set(0.0);
		 }
           
		 		 
		 ++RFSegOutIterator;
		 ++inputIterator;
		 it++;
	}
	
	
	ImagePointer RFSegOutImage2=ImageType::New();
	RFSegOutImage2=RFSegOutImage;
	NeighborhoodIteratorType RFSegOutIterator2(radius, RFSegOutImage2, outRegion);
	
	RFSegOutIterator.GoToBegin();
	//adjust using counts
	
	
	while(!RFSegOutIterator2.IsAtEnd())
	{
	   
		int count = 0;
		
		for(int i=0; i<c; i++){ //don't count centre pixel
			if(RFSegOutIterator2.GetPixel(i) == 1.0 & i != (c/2) ){++count;}
					
		}	
		
		if(count< min_neighbour){
			RFSegOutIterator.Set(0.0);
		}else{
			RFSegOutIterator.Set(RFSegOutIterator2.GetCenterPixel());
			}
			
  	   
		
		++RFSegOutIterator;
		++RFSegOutIterator2;
	}
		
  
   NiftiWriter(RFSegOutImage,rfSegOutFilename.c_str()); 

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()


//FOR TEST: 'testingSamples' IS PASSED TO THIS METHOD ONLY WHEN TESTING & DEBUGGING. IN REAL CASES, THESE SAMPLES WILL BE EXTRACTED FROM THE INPUT IMAGES.
//ImagePointer ClassifyWMHs(ImagePointer WMModStripImg,CvRTrees* RFRegressionModel, int featuresCount,char *rfSegOutFilename, Mat testingSamples)
ImagePointer ClassifyWMHsM(ImagePointer WMModStripImg, std::string rfSegOutFilename, int min_neighbour, double out_thresh_const, int max_iter=100, double tol = 0.0001)
{

	
	std::cout << "Performing WMH segmentation with M-estimator..." << std::endl;
	
	MarginateImage(WMModStripImg,5);                        
	//Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.
	// kept in to keep consistent with older RF model
	
	
	//creating an output image (i.e. a segmentation image) out of the prediction results.
	WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
	ImagePointer RFSegOutImage=ImageType::New();
	
	ImageType::IndexType outStartIdx;
	outStartIdx.Fill(0);
	
	ImageType::SizeType outSize=WMModStripImg->GetLargestPossibleRegion().GetSize();
	ImageType::RegionType outRegion;
	outRegion.SetSize(outSize);
	outRegion.SetIndex(outStartIdx);
	
	RFSegOutImage->SetRegions(outRegion);
	RFSegOutImage->SetDirection(WMModStripImg->GetDirection());   //e.g. left-right Anterior-Posterior Sagittal-...
	RFSegOutImage->SetSpacing(WMModStripImg->GetSpacing());      //e.g. 2mm*2mm*2mm
	RFSegOutImage->SetOrigin(WMModStripImg->GetOrigin());
	RFSegOutImage->Allocate();
	
	     
	NeighborhoodIteratorType::RadiusType radius;
	radius.Fill(1);
	
	// set up iterators
	itk::ImageRegionIterator<ImageType> RFSegOutIterator(RFSegOutImage, outRegion);
		
	NeighborhoodIteratorType inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
	
	//initialise vectors
	std::vector<double> y;
	std::vector<double> zeros;
	
	//model fit
	// robust fit normal dist with M estimator
  	
  	//put data into std::vector
	while(!inputIterator.IsAtEnd())
	{
		y.push_back(inputIterator.GetCenterPixel());
		if(inputIterator.GetCenterPixel()>0.0){
			zeros.push_back(1.0);
		}else{
			zeros.push_back(0.0);
		}
		++inputIterator;
	}
	
	
	
	vecPair model = fitMest(y, zeros,  tol, max_iter);
		
	double mu = model.second.at(1);
	double sigma = model.second.at(0);   
	
	vector<double>::iterator it; 
	
    it=model.first.begin();
    
	//malahabnois dist cutoff as in van leemput 99
	double outly = -2*std::log(out_thresh_const*std::sqrt(2*3.141593*sigma)); 
    
    
   
	inputIterator.GoToBegin();
	int c = (inputIterator.Size());
	int mid = (inputIterator.Size()) / 2;
	
	
	
	
	
   while(!inputIterator.IsAtEnd())
   {
		double acc = 0.0;
		double neighbourhood_mean = 0.0;
		int non_zero = 0;
		
		for(int i=0; i<c; i++){		
			double temp  = inputIterator.GetPixel(i); 
			
			if(i != (c/2))
			{
				acc += temp;
				if(temp>0){++non_zero;}
			}
		
		}	
		
		if(non_zero>0.){
			neighbourhood_mean = acc/non_zero;			       
        }else{
			neighbourhood_mean = 0.;	
		}
           
       
        
         
		

         if(neighbourhood_mean > mu & *it > outly){
			RFSegOutIterator.Set(1.0);
			std::cout << "local mean " << neighbourhood_mean << " mean "<< mu << " dist "<< *it << " cutoff "<< outly  << std::endl;
			
		 }else{
			RFSegOutIterator.Set(0.0);
		 }
           
		 		 
		 ++RFSegOutIterator;
		 ++inputIterator;
		 it++;
	}
		
	
	ImagePointer RFSegOutImage2=ImageType::New();
	RFSegOutImage2=RFSegOutImage;
	NeighborhoodIteratorType RFSegOutIterator2(radius, RFSegOutImage2, outRegion);
	
	RFSegOutIterator.GoToBegin();
	
	//adjust using counts
	while(!RFSegOutIterator2.IsAtEnd())
	{
	   
		int count = 0;
		
		for(int i=0; i<c; i++){
			if(RFSegOutIterator2.GetPixel(i) == 1.0 & i != (c/2)){++count;}
		}	
		
		if(count< min_neighbour){
			RFSegOutIterator.Set(0.0);
		}else{
			RFSegOutIterator.Set(RFSegOutIterator2.GetCenterPixel());
		}
			
  	   
		
		++RFSegOutIterator;
		++RFSegOutIterator2;
	}
		
  
   NiftiWriter(RFSegOutImage,rfSegOutFilename.c_str()); 

   std::cout << "Done WMH segmentation successfully." << std::endl;
   return RFSegOutImage;
}//end of ClassifyWMHs()


vecPair fitMest(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter){

	std::cout<< "observation length:  " << y.size() << " pixels" <<   std::endl;
	
  	//double z_sum = std::accumulate(zeros.begin(), zeros.end(), 0.0);
  	double y_sum = std::accumulate(y.begin(), y.end(), 0.0);
	//double y_mean = y_sum / y.size();
	
	//allocate vectors
	std::vector<double> w = zeros;  
	std::vector<double> d(y.size(), 0.0);
	std::vector<double> wy_prod(y.size(), 0.0);  
	std::vector<double> ymu_minus(y.size(), 0.0);
	std::vector<double> ymu_minus2(y.size(), 0.0);
	std::vector<double> sq_sum(y.size(), 0.0);
	std::vector<double> sq_sum_w(y.size(), 0.0);
	std::vector<double> w2(y.size(), 0.0); 
	std::vector<double> diff_w(y.size(), 0.0);
	std::vector<double> outliness(y.size(), 0.0);
	std::vector<double> log_w(y.size(), 0.0);
	
	//allocate double
	double mu = 0.0;
	double sigma = 0.0;
	double wy_sum = 0.0;
	double w_sum = 0.0;
	double w_sq_sum = 0.0;
	double v_sum = 0.0;
	double diff = 1.0;
	
	//allocated int
	int k =0;
	
	
	//iterative fit for M-est
	while(diff>tol & k <max_iter ){
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>());
		
		std::transform(y.begin(), y.end(), w.begin(), wy_prod.begin(), std::multiplies<double>()); 
		wy_sum = std::accumulate(wy_prod.begin(), wy_prod.end(), 0.0);
		w_sum = std::accumulate(w.begin(), w.end(), 0.0);
		
		mu = y_sum/w_sum;
		
		std::transform(y.begin(), y.end(), ymu_minus.begin(), [mu](double x) { return x - mu; });
			
		std::transform(ymu_minus.begin(), ymu_minus.end(), ymu_minus.begin(), sq_sum.begin(), std::multiplies<double>()); 
		std::transform(sq_sum.begin(), sq_sum.end(), w.begin(), sq_sum_w.begin(), std::multiplies<double>()); 
		sigma =  std::accumulate(sq_sum_w.begin(), sq_sum_w.end(), 0.0) / w_sum; //sigma is variance not sd
		
		std::transform(sq_sum.begin(), sq_sum.end(), d.begin(), [sigma](double x) { return std::sqrt(x / sigma); });
		
		w2 = w;
		std::transform(d.begin(), d.end(), w.begin(), phi);
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>()); //ignores pixels with value = 0
		
		std::transform(w.begin(), w.end(), w2.begin(), diff_w.begin(), [](double x, double y) {return std::abs(x-y);}); 
		
		diff = std::accumulate(diff_w.begin(), diff_w.end(), 0.0);
		++k;
	
	}
	
	std::transform(d.begin(), d.end(), zeros.begin(), d.begin(), std::multiplies<double>());  //ignores pixels with value = 0
	
	std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " diff " << diff << std::endl;
	
	std::vector<double> params;
	params.push_back(sigma);
	params.push_back(mu);
	
	
	vecPair returnValues = vecPair(d, params);
	
	return(returnValues);
}



vecPair fitTdist(std::vector<double> y, std::vector<double> zeros,  double tol, int max_iter, double a, double b){

	std::cout<< "observation length:  " << y.size() << " pixels" <<   std::endl;
	
  	double z_sum = std::accumulate(zeros.begin(), zeros.end(), 0.0);
  	double y_sum = std::accumulate(y.begin(), y.end(), 0.0);
	//double y_mean = y_sum / y.size();
	
	//allocate vectors
	std::vector<double> w = zeros;  
	std::vector<double> d(y.size(), 0.0);
	std::vector<double> wy_prod(y.size(), 0.0);  
	std::vector<double> ymu_minus(y.size(), 0.0);
	std::vector<double> ymu_minus2(y.size(), 0.0);
	std::vector<double> sq_sum(y.size(), 0.0);
	std::vector<double> sq_sum_w(y.size(), 0.0);
	std::vector<double> w2(y.size(), 0.0); 
	std::vector<double> diff_w(y.size(), 0.0);
	std::vector<double> outliness(y.size(), 0.0);
	std::vector<double> log_w(y.size(), 0.0);
	
	//allocate double
	double mu = 0.0;
	double sigma = 0.0;
	double wy_sum = 0.0;
	double w_sum = 0.0;
	double w_sq_sum = 0.0;
	double v_sum = 0.0;
	double diff = 1.0;
	double v = 4.0; // initial guess for degrees of freedom, could be user param but actually not very important, range (a,b) more important.
	
	//allocated int
	int k =0;
	
	while(diff>tol & k <max_iter ){
		    
		boost::uintmax_t df_max_iter=500; // boost solver params, could be user params but relatively unimportant
		tools::eps_tolerance<double> tol(30);
		
		std::transform(y.begin(), y.end(), w.begin(), wy_prod.begin(), std::multiplies<double>()); 
		wy_sum = std::accumulate(wy_prod.begin(), wy_prod.end(), 0.0);
		w_sum = std::accumulate(w.begin(), w.end(), 0.0);
		mu = y_sum/z_sum;
		
		std::transform(y.begin(), y.end(), ymu_minus.begin(), [mu](double x) { return x - mu; });
			
		
		std::transform(ymu_minus.begin(), ymu_minus.end(), ymu_minus.begin(), sq_sum.begin(), std::multiplies<double>()); 
		std::transform(sq_sum.begin(), sq_sum.end(), w.begin(), sq_sum_w.begin(), std::multiplies<double>()); 
		sigma =  std::accumulate(sq_sum_w.begin(), sq_sum_w.end(), 0.0) / z_sum; //sigma is variance not sd
		
				
				
		std::transform(w.begin(),w.end(), log_w.begin(), [](double x) { return (std::log(x) - x); });
		std::transform(log_w.begin(), log_w.end(), log_w.begin(), [](double x) { return isinf(x) ? 0.0 : x  ; }); //ignores pixels with value = 0
		v_sum = std::accumulate(log_w.begin(), log_w.end(), 0.0)/ z_sum;
		
		if(k !=0){
			df_eq_func rootFun = df_eq_func(v_sum);
			std::pair<double, double>  r1= tools::bisect(rootFun, a, b, tol, df_max_iter);
			v = (r1.first + r1.second)/2.0; 
		}
		
		
		std::transform(sq_sum.begin(), sq_sum.end(), d.begin(), [sigma](double x) { return std::sqrt(x / sigma); });
		w2 = w;	
		
		
		std::transform(d.begin(), d.end(), w.begin(), [v](double x) { return ((v+1) / (v+x*x));});
		std::transform(w.begin(), w.end(), zeros.begin(), w.begin(), std::multiplies<double>()); //ignores pixels with value = 0
		
		std::transform(w.begin(), w.end(), w2.begin(), diff_w.begin(), [](double x, double y) {return std::abs(x-y);}); 
		
		diff = std::accumulate(diff_w.begin(), diff_w.end(), 0.0);
		
		//std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " v " << v << " diff " << diff << std::endl;
		
		++k;
	
	}
    std::transform(d.begin(), d.end(), zeros.begin(), d.begin(), std::multiplies<double>());  //ignores pixels with value = 0

	std::cout<< "iter " << k << " mu " << mu << " sigma " << sigma << " v " << v << " diff " << diff << std::endl;	

	std::vector<double> params;
	params.push_back(v);
	params.push_back(sigma);
	params.push_back(mu);
	
	vecPair returnValues = vecPair(d, params);
	return(returnValues);
}






void QuantifyWMHs(float pmapCut, ImagePointer pmapImg, std::string ventricleFilename, std::string outputFilename)
{
   std::cout << "Quantifying WMH segmentations..." << std::endl;

   //voxelResolution has been hard-coded as '0.5' in the W2MHS toolbox code ('V_res').
   //float voxelRes=pmapImg->GetSpacing()[0];
   float voxelRes=pmapImg->GetSpacing()[0]*pmapImg->GetSpacing()[1]*pmapImg->GetSpacing()[2];
   int distDP=8;   //According to the definition of "Periventricular Region" in "“Anatomical mapping of white matter hyper- intensities (wmh) exploring the relationships between periventricular wmh, deep wmh, and total wmh burden, 2005", voxels closer than 8mm to ventricle are considered as 'Periventricular' regions.
   int k=1;   // Note: "EV calculates the hyperintense voxel count 'weighted' by the corresponding likelihood/probability, where 'k' controls the degree of weight (in formula (2))." ... 'k' is called 'gamma' in the W2MHS toolbox code.

   ImagePointer thresholdedPmap=ThresholdImage(pmapImg,pmapCut,1);
   float EV=voxelRes * SumImageVoxels(PowerImageToConst(thresholdedPmap,k));

   //The following block differentiates the Periventricular from Deep hyperintensities.
   ImagePointer ventricle=NiftiReader(ventricleFilename.c_str());

   //creating a 'Ball' structuring element for dilating the ventricle area. (It can also be created using 'itk::BinaryBallStructuringElement'.)
   /*
    * "A Neighborhood has an N-dimensional radius. The radius is defined separately for each dimension as the number of pixels that the neighborhood extends outward from the center pixel.
    * For example, a 2D Neighborhood object with a radius of 2x3 has sides of length 5x7. However, in the case of balls and annuli, this definition is slightly different from the parametric
    * definition of those objects. For example, an ellipse of radius 2x3 has a diameter of 4x6, not 5x7. To have a diameter of 5x7, the radius would need to increase by 0.5 in each dimension.
    * Thus, the "radius" of the neighborhood and the "radius" of the object should be distinguished.
    * To accomplish this, the "ball" and "annulus" structuring elements have an optional flag called "radiusIsParametric" (off by default). Setting this flag to true will use the parametric definition
    * of the object and will generate structuring elements with more accurate areas, which can be especially important when morphological operations are intended to remove or retain objects of particular sizes.
    * When the mode is turned off (default), the radius is the same, but the object diameter is set to (radius*2)+1, which is the size of the neighborhood region. Thus, the original ball and annulus structuring
    * elements have a systematic bias in the radius of +0.5 voxels in each dimension relative to the parametric definition of the radius. Thus, we recommend turning this mode on for more accurate structuring elements,
    * but this mode is turned off by default for backward compatibility."
    */
   typedef itk::FlatStructuringElement<3> StructuringElement3DType;      //3 is the image dimension
   StructuringElement3DType::RadiusType radius3D;
   radius3D.Fill(distDP/voxelRes);      //"voxels closer than 8mm to ventricle are considered as 'DeepPeriventricular' regions.". Thus, when for example the voxel resolution of the image is 2mm, it means that those voxels that are within 8mm/2mm=4 voxels away from the ventricle should be considered as periventricular.
   StructuringElement3DType structuringElem=StructuringElement3DType::Ball(radius3D);
//   structuringElem.RadiusIsParametricOn();   //and "structuringElem.SetRadiusIsParametric(true)" make no difference in calculations, decpite what's been claimed (see above explanation)!

   //dilating the ventricle area by the 'Ball' structuring element
   float pEV=0;
   float dEV=0;
   try
   {
      typedef itk::BinaryDilateImageFilter<ImageType, ImageType, StructuringElement3DType> BinaryDilateImageFilterType;
      BinaryDilateImageFilterType::Pointer dilateFilter = BinaryDilateImageFilterType::New();
      dilateFilter->SetInput(ventricle);
      dilateFilter->SetKernel(structuringElem);
      dilateFilter->SetForegroundValue(1);   //Note: "Set the value in the image to consider as "foreground". Defaults to maximum value of PixelType."
      dilateFilter->SetBackgroundValue(0);   //      Thus, these two lines are necessary, as the index type is set to float.
      ImagePointer dilatedVentricle=dilateFilter->GetOutput();
      dilatedVentricle->Update();

      //separating Deep and Periventricular areas in pmap.
      ImagePointer periventricularPmap=MultiplyTwoImages(thresholdedPmap,dilatedVentricle);
      ImagePointer deepPmap=MultiplyTwoImages(thresholdedPmap,InvertImage(dilatedVentricle,1));
      //calculating pEV and dEV
      pEV=voxelRes * SumImageVoxels(PowerImageToConst(periventricularPmap,k));
      dEV=voxelRes * SumImageVoxels(PowerImageToConst(deepPmap,k));
   }
   catch(itk::ExceptionObject &)
   {
      //if, for example, the 'ventricle' image hasn't been uploaded for some reason, as exception happens.
      std::cerr << "Failed to Quantify WMH segmentations!" << std::endl;
   }

   //saving the outputs of WMH burden calculation into a .txt file.
   try
   {
      std::ofstream EVQuantFile;
      EVQuantFile.open(outputFilename,std::ios::app);
      EVQuantFile << "EV= " << EV << "\n";
      EVQuantFile << "EV-Deep= " << dEV << "\n";
      EVQuantFile << "EV-Periventricular= " << pEV << "\n";
      EVQuantFile.close();
      std::cout << "Quantification of WMH segmentations is saved into: " << outputFilename << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      //if, for exampleImagePointer MultiplyTwoImages(ImagePointer input1,ImagePointer input2);, user doesn't have write permission to the specified file/folder, an exception happens.
      std::cerr << "Failed to save the quantification of WMH segmentations into the file: " << outputFilename << std::endl;
   }  
}//end of QuantifyWMHs()




ImagePointer NiftiReader(char * inputFile)
{
   typedef itk::ImageFileReader<ImageType> ImageReaderType;
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();
   ImageReaderType::Pointer imageReader=ImageReaderType::New();
   ImagePointer output;
   try
   {
      niftiIO->SetFileName(inputFile);
      niftiIO->ReadImageInformation();

      imageReader->SetImageIO(niftiIO);
      imageReader->SetFileName(niftiIO->GetFileName());
      output=imageReader->GetOutput();
      output->Update();
      std::cout << "Successfully read: " << inputFile << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to read: " << inputFile << std::endl;
   }
   return output;
}//end of NiftiReader()

ImagePointer NiftiReader(std::string inputFile)
{
   typedef itk::ImageFileReader<ImageType> ImageReaderType;
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();
   ImageReaderType::Pointer imageReader=ImageReaderType::New();
   ImagePointer output;
   try
   {
      niftiIO->SetFileName(inputFile);
      niftiIO->ReadImageInformation();

      imageReader->SetImageIO(niftiIO);
      imageReader->SetFileName(niftiIO->GetFileName());
      output=imageReader->GetOutput();
      output->Update();
      std::cout << "Successfully read: " << inputFile << std::endl;
   }
   catch(itk::ExceptionObject &)
   {
      std::cerr << "Failed to read: " << inputFile << std::endl;
   }
   return output;
}//end of NiftiReader()

bool NiftiWriter(ImagePointer input,std::string outputFile)
{
   typedef itk::ImageFileWriter<ImageType> imageWriterType;
   imageWriterType::Pointer imageWriterPointer =imageWriterType::New();
   itk::NiftiImageIO::Pointer niftiIO=itk::NiftiImageIO::New();

   try
   {
      //Set the output filename
      imageWriterPointer->SetFileName(outputFile);
      //Set input image to the writer.
      imageWriterPointer->SetInput(input);
      //Determine file type and instantiate appropriate ImageIO class if not explicitly stated with SetImageIO, then write to disk.
      imageWriterPointer->SetImageIO(niftiIO);

      imageWriterPointer->Write();
      std::cout << "Successfully saved image: " << outputFile << std::endl;
      return true;
   }
   catch ( itk::ExceptionObject & ex )
   {
      std::string message;
      message = "Problem found while saving image ";
      message += outputFile;
      message += "\n";
      message += ex.GetLocation();
      message += "\n";
      message += ex.GetDescription();
      std::cerr << message << std::endl;
      return false;
   }
}//end of NiftiWriter()


Mat getLocationVector(ImagePointer WMModStripImg)
{
   	
   MarginateImage(WMModStripImg,5);                        //Patch width is 5 voxels. Thus, a margin of size 5 voxels will be discarded to avoid any problems when doing image convolution with kernels.


   WMModStripImg->SetRequestedRegionToLargestPossibleRegion();
   
 
  ImageType::SizeType radius;
  radius[0] = 1;
  radius[1] = 1;
  radius[2] = 1;
  
  itk::NeighborhoodIterator<ImageType> inputIterator(radius, WMModStripImg, WMModStripImg->GetRequestedRegion());
 
   
   
   
   Mat outMat = Mat(0, 5, CV_32FC1);  
   
     
   std::cout << "begin loop" << std::endl;
   
   while(!inputIterator.IsAtEnd()){
		//if(inputIterator.Get() > threshold)                  //the voxels that are not within the threshold (i.e. 0.3 the input image histogram) will be discarded and the corresponding value of 0 will be saved in the segmentation image.
         //{

			//std::cout <<  inputIterator.Get() << "  " << inputIteratorBRAVO.Get()  << std::endl;
			Mat Temp = Mat(1, 5, CV_32FC1);
			Temp.at<float>(0,0)=inputIterator.GetCenterPixel();
			
			
			ImageType::IndexType tempIndex = inputIterator.GetIndex();
			double acc = 0.0;
			double neighbourhood_mean = 0.0;
			int non_zero = 0;
			int c = (inputIterator.Size());
			
			for(int i=0; i<c; i++){		
				double temp  = inputIterator.GetPixel(i);
				
				if(i != (c/2))
				{
					acc += temp;
					if(temp>0){++non_zero;}
				}
			
			}	
			if(non_zero == 0) {
				neighbourhood_mean = 0.0;
			}else{
				neighbourhood_mean = acc/non_zero;
			}
				
			Temp.at<float>(0,1)=neighbourhood_mean;
					
			
			
			Temp.at<float>(0,2)=tempIndex[0];
			Temp.at<float>(0,3)=tempIndex[1];
			Temp.at<float>(0,4)=tempIndex[2];
			
			//std::cout << tempIndex[0] << " " << tempIndex[0] << " " << tempIndex[2] <<  std::endl;;
				    
			outMat.push_back(Temp);
			++inputIterator;
			
	   //}
   }

   

   std::cout << "Done feature production! "<< std::endl;
   return outMat;
}


void MarginateImage(ImagePointer input,int marginWidth)
{
   //Note: itk::CropImageFilter hasn't been used for this purpose, as it throws away the cropped margin (which is not what we need here).
   //this function receives an Image Pointer. So, the changes that are applied to the image that exists at the specified pointer will remain (even) after returning from this function.
   //this function is used in W2MHS toolbox (probably) to avoid falling outside of the image boundaries while extracting patches for each voxel.
   //If this is the case, there should also be no problem if we marginate the image by 1/2 of the patch size. This is because each patch around each voxel has a "radius" equal to 1/2 of the patch size.
   //TODO: (issue120302) it may not really be necessary to set this margin to 0. Instead, we can probably create a region with the size of this margin
   //      and then create an imageIterator which only iterates inside this margin of the input image ---> itk::ImageRegionIterator<ImageType> imageIterator(input, marginRegion);
   input->SetRequestedRegionToLargestPossibleRegion();
   itk::ImageRegionIterator<ImageType> imageIterator(input, input->GetRequestedRegion());   //for iteration through the whole image
   while(!imageIterator.IsAtEnd())
   {
      ImageType::IndexType currentIdx=imageIterator.GetIndex();
      if(currentIdx[0] < marginWidth || currentIdx[1] < marginWidth || currentIdx[2] < marginWidth
            || currentIdx[0] > (input->GetRequestedRegion().GetSize()[0]-1-marginWidth)
            || currentIdx[1] > (input->GetRequestedRegion().GetSize()[1]-1-marginWidth)
            || currentIdx[2] > (input->GetRequestedRegion().GetSize()[2]-1-marginWidth))
         imageIterator.Set(0);
      ++imageIterator;
   }
}//end of MarginateImage()


ImagePointer MultiplyTwoImages(ImagePointer input1,ImagePointer input2)
{
   //Note: Pixel-wise multiplication of two images

   typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyImageFilterType;
   MultiplyImageFilterType::Pointer multiplyImageFilter = MultiplyImageFilterType::New();
   multiplyImageFilter->SetInput1(input1);
   multiplyImageFilter->SetInput2(input2);
   ImagePointer output=multiplyImageFilter->GetOutput();
   output->Update();
   return output;
}//MultiplyTwoImages()



ImagePointer ThresholdImage(ImagePointer input, float lowerThreshold, float upperThreshold)
{
   /* Note:
    * "ThresholdImageFilter sets image values to a user-specified "outside" value (by default, "black") if the
    * image values are below, above, or between simple threshold values."
    */
   typedef itk::ThresholdImageFilter<ImageType> ThresholdImageFilterType;
   ThresholdImageFilterType::Pointer thresholdFilter = ThresholdImageFilterType::New();
   thresholdFilter->SetInput(input);
   thresholdFilter->ThresholdOutside(lowerThreshold, upperThreshold);   //other options are: ThresholdAbove() and ThresholdBelow()   Note that "pixels equal to the threshold value are NOT set to OutsideValue in any of these methods"
   thresholdFilter->SetOutsideValue(0);

   ImagePointer output=thresholdFilter->GetOutput();
   output->Update();

   return output;
}//end of ThresholdImage()


float SumImageVoxels(ImagePointer input)
{

   typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
   StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
   statisticsImageFilter->SetInput(input);
   statisticsImageFilter->Update();

   return statisticsImageFilter->GetSum();
}//end of SumImageVoxels()

//adds suffix and puts extensiuon back in place
std::string renamer(std::string baseName, std::string suffix){
	
	size_t lastindex = baseName.find_last_of("."); 
	std::string rawname = baseName.substr(0, lastindex);
	std::string exten = baseName.substr(baseName.find_last_of(".") + 1);
	
	std::string newName = rawname + suffix +"."+ exten;
	
	return newName;

}

ImagePointer PowerImageToConst(ImagePointer input, float constPower)
{
   if(constPower == 1 )
      return input;
   else
   {
      typedef itk::PowImageFilter<ImageType,ImageType,ImageType> PowImageFilterType;
      PowImageFilterType::Pointer powImageFilter = PowImageFilterType::New();
      powImageFilter->SetInput1(input);
      powImageFilter->SetConstant2(constPower);   //Note: "Additionally, this filter can be used to raise every pixel of an image to a power of a constant by using SetConstant2()."
      ImagePointer output=powImageFilter->GetOutput();
      output->Update();

      return output;
   }
}//end of PowerImageToConst()



ImagePointer InvertImage(ImagePointer input, int maximum)
{
   //Note: InvertIntensityImageFilter inverts intensity of pixels by subtracting pixel value to a maximum value. The maximum value can be set with SetMaximum and defaults the maximum of input pixel type.
   //      This function may also be implemented using the itk::BinaryNotImageFilter when the input image is a binary image
   typedef itk::InvertIntensityImageFilter<ImageType> InvertIntensityImageFilterType;

   InvertIntensityImageFilterType::Pointer invertIntensityFilter = InvertIntensityImageFilterType::New();
   invertIntensityFilter->SetInput(input);
   invertIntensityFilter->SetMaximum(maximum);
   ImagePointer output=invertIntensityFilter->GetOutput();
   output->Update();
   return output;
}//end of InvertImage()
