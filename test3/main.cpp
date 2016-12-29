/* decimator.cpp */

// Polyphase decimation filter.
//
// Convert an oversampled audio stream to non-oversampled.  Uses a
// windowed sinc FIR filter w/ Blackman window to control aliasing.
// Christian Floisand's 'blog explains it very well.
//
// This version has a very simple main processing loop (the decimate
// method) which vectorizes easily.
//
// Refs:
//   https://christianfloisand.wordpress.com/2012/12/05/audio-resampling-part-1/
//   https://christianfloisand.wordpress.com/2013/01/28/audio-resampling-part-2/
//   http://www.dspguide.com/ch16.htm
//   http://en.wikipedia.org/wiki/Window_function#Blackman_windows

#define _USE_MATH_DEFINES


#include <math.h>
#include <assert.h>
#include <cstddef>
#include <iostream>
#include <cstdio>
#include <complex>
#include <map>

//using namespace Aquila;
//below by_YJY
#include <algorithm>
#include <cmath>
#include <time.h>
#include <valarray>
#include <typeinfo>
const double PI = 3.141592653589793238460;
typedef std::complex<float> Complex;
typedef std::valarray<Complex> CArray;
const double LN_2 = 0.69314718055994530941723212145818;
typedef unsigned int fftWiCacheKeyType;
typedef std::map<fftWiCacheKeyType, Complex**> fftWiCacheType;

//end by_YJY

class Decimator {

    public:
        Decimator();
        ~Decimator();

        void   initialize(double   decimatedSampleRate,
                double   passFrequency,
                unsigned oversampleRatio);

        double oversampleRate()  const { return mOversampleRate; }
        int    oversampleRatio() const { return mRatio; }

        void   decimate(float *in, float *out, size_t outCount);
        // N.B., input must have (ratio * outCount) samples.

    private:
        double mDecimatedSampleRate;
        double mOversampleRate;
        int    mRatio;              // oversample ratio
        float *mKernel;
        size_t mKernelSize;
        float *mShift;              // shift register
        size_t mCursor;

};



Decimator::Decimator()
    : mKernel(NULL),
    mShift(NULL)
{}

Decimator::~Decimator()
{
    delete[] mKernel;
    delete[] mShift;
}

void Decimator::initialize(double   decimatedSampleRate,
        double   passFrequency,
        unsigned oversampleRatio)
{
    mDecimatedSampleRate = decimatedSampleRate;
    mRatio = oversampleRatio;
    mOversampleRate = decimatedSampleRate * oversampleRatio;

    double NyquistFreq = decimatedSampleRate / 2;
    assert(passFrequency < NyquistFreq);

    // See DSP Guide.
    double Fc = (NyquistFreq + passFrequency) / 2 / mOversampleRate;
    double BW = (NyquistFreq - passFrequency) / mOversampleRate;
    int M = ceil(4 / BW);
    if (M % 2) M++;
    size_t activeKernelSize = M + 1;
    size_t inactiveSize = mRatio - activeKernelSize % mRatio;
    mKernelSize = activeKernelSize + inactiveSize;

    // DSP Guide uses approx. values.  Got these from Wikipedia.
    double a0 = 7938. / 18608., a1 = 9240. / 18608., a2 = 1430. / 18608.;

    // Allocate and initialize the FIR filter kernel.
    delete[] mKernel;
    mKernel = new float[mKernelSize];
    double gain = 0;
    for (size_t i = 0; i < inactiveSize; i++)
        mKernel[i] = 0;
    for (int i = 0; i < activeKernelSize; i++) {
        double y;
        if (i == M / 2)
            y = 2 * M_PI * Fc;
        else
            y = (sin(2 * M_PI * Fc * (i - M / 2)) / (i - M / 2) *
                    (a0 - a1 * cos(2 * M_PI * i / M) + a2 * cos(4 * M_PI / M)));
        gain += y;
        mKernel[inactiveSize + i] = y;
    }

    // Adjust the kernel for unity gain.
    float inv_gain = 1 / gain;
    for (size_t i = inactiveSize; i < mKernelSize; i++)
        mKernel[i] *= inv_gain;

    // Allocate and clear the shift register.
    delete[] mShift;
    mShift = new float[mKernelSize];
    for (size_t i = 0; i < mKernelSize; i++)
        mShift[i] = 0;
    mCursor = 0;
}

// The filter kernel is linear.  Coefficients for oldest samples
// are on the left; newest on the right.
//
// The shift register is circular.  Oldest samples are at cursor;
// newest are just left of cursor.
//
// We have to do the multiply-accumulate in two pieces.
//
//  Kernel
//  +------------+----------------+
//  | 0 .. n-c-1 |   n-c .. n-1   |
//  +------------+----------------+
//   ^            ^                ^
//   0            n-c              n
//
//  Shift Register
//  +----------------+------------+
//  |   n-c .. n-1   | 0 .. n-c-1 |
//  +----------------+------------+
//   ^                ^            ^
//   mShift           shiftp       n

void Decimator::decimate(float *in, float *out, size_t outCount)
{
    assert(!(mCursor % mRatio));
    assert(mCursor < mKernelSize);
    size_t cursor = mCursor;
    float *inp = in;
    float *shiftp = mShift + cursor;
    for (size_t i = 0; i < outCount; i++) {

        // Insert mRatio input samples at cursor.
        for (size_t j = 0; j < mRatio; j++)
            *shiftp++ = *inp++;
        if ((cursor += mRatio) == mKernelSize) {
            cursor = 0;
            shiftp = mShift;
        }

        // Calculate one output sample.
        double acc = 0;
        size_t size0 = mKernelSize - cursor;
        size_t size1 = cursor;
        const float *kernel1 = mKernel + size0;
        for (size_t j = 0; j < size0; j++)
            acc += shiftp[j] * mKernel[j];
        for (size_t j = 0; j < size1; j++)
            acc += mShift[j] * kernel1[j];
        out[i] = acc;
    }
    mCursor = cursor;
}

/*by internet*/
// Cooley–Tukey FFT (in-place)
void fft(CArray& x)
{
    //double i;
    const size_t N = x.size();
    if (N <= 1) return;
    //if(modf(log2(x.size()),&i)!=0){
    //x.resize(pow(2,ceil(log2(x.size()))));}
    //std::cout<<"origin : "<<N<<" , zeropadding : "<<x.size()<<std::endl;
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];

    // conquer
    fft(even);
    fft(odd);

    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar((float)1.0, (float)(-2 * PI * k / N)) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
   // x.resize(N);
}

// inverse fft (in-place)
void ifft(CArray& x)
{
    // conjugate the complex numbers
    x = x.apply(std::conj);

    // forward fft
    fft( x );

    // conjugate the complex numbers again
    x = x.apply(std::conj);

    // scale the numbers
    x /= x.size();
}

void preprocessing(CArray & out, float (*finaloutput)[26460])
{
    //float *inp = out;
    int bandlimits[7] = { 0,200,400,800,1600,3200,4096 };
    int nbands = 6;
    //filterbank
    CArray dft = out;
    fft(dft);
    int n=dft.size();
    int bl[nbands];
    int br[nbands];
    for(int i=0;i<nbands;++i)
    {
        bl[i]=floor(bandlimits[i]*n*0.5/bandlimits[nbands])+1;
        br[i]=floor(bandlimits[i+1]*n*0.5/bandlimits[nbands]);
    }
    br[nbands-1] = floor(n*0.5);
   /* 
    for(int i=0;i<nbands;++i)
    {
        std::cout<<"bl : "<<bl[i]<<" , br : "<<br[i]<<std::endl;
    }*/
    CArray output1[nbands];
    for(int i=0;i<nbands;++i)
    {
        output1[i].resize(n);
        for(int j=bl[i]-1;j<br[i];++j)
        {
            output1[i][j]=dft[j];
            output1[i][n-j-1]=dft[n-j-1];
        }
       // std::cout<<"i = "<<i<<" , left : "<<bl[i]-1<<"~"<<br[i]-1<<" , right : "<<n-bl[i]<<"~"<<n-br[i]<<std::endl;
    }
    output1[0][0]=Complex();
    FILE *fp20 = fopen("filterbankTEST.txt", "w+");
    for (int i = 0; i < nbands; i++) {
         for(int j=0;j<n;++j)
         { 
        if(output1[i][j].imag() >= 0)
          fprintf(fp20, "%f+%fi ", output1[i][j].real(),output1[i][j].imag());
         else
             fprintf(fp20,"%f%f ",output1[i][j].real(),output1[i][j].imag());
        }
         fprintf(fp20,"\n");
    }
    fclose(fp20);
    //hwindow
    float winlength=0.4f;
    float hannlen=(float)(2*bandlimits[6]*winlength);
    CArray hann(n);
    for(int i=0;i<floor(hannlen);++i)
    {
        hann[i] =Complex((float)(cos((i+1)*M_PI/hannlen/2)*cos((i+1)*M_PI/hannlen/2)));
    }
    CArray wave[nbands]=output1;
    for(int i=0;i<nbands;++i)
    {
            ifft(output1[i]);
            for(int j=0;j<n;++j)
            {
                wave[i][j]=Complex(output1[i][j].real(),0);
            }
    }
    CArray freq[nbands]=output1;
    for(int i=0;i<nbands;++i)
    {
        for(int j=0;j<n;++j)
        {
            if(wave[i][j].real()<0)
            {
                wave[i][j].real()*=-1;
            }
        }
        fft(wave[i]);
        for(int j=0;j<n;++j)
        {
        freq[i][j]=wave[i][j];
        }
    }
    CArray filtered[nbands]=output1;
    float output2[nbands][n]={0};
    for(int i=0;i<nbands;++i)
    {
        fft(hann);
        for(int j=0;j<n;++j)
        {
            filtered[i][j]=freq[i][j]*hann[j];
        }
        ifft(filtered[i]);
        for(int j=0;j<n;++j)
        {
        output2[i][j]=filtered[i][j].real();
        //std::cout<<output2[i][j]<<" ";
        }
        //std::cout<<"\n";*/
    }
    FILE *fp3 = fopen("testHwindow.txt", "w+");
    for (int i = 0; i < nbands; i++) {
        for(int j=0;j<n;++j)
        {
            fprintf(fp3,"%f ",output2[i][j]);
        }
        fprintf(fp3,"\n");
    }
    fclose(fp3);
    //diffrect
    //final[nbands][n]={0};
    for(int i=0;i<nbands;++i)
    {
        for(int j=0;j<n;++j)
        {
            float d = output2[i][j]-output2[i][j-1];
            //std::cout<<output2[i][j]<<", "<<output2[i][j-1]<<std::endl;
            if(d>0)
            {
                finaloutput[i][j]=d;
                //std::cout<<finaloutput[i][j]<<"dsfs";
            }
            //std::cout<<finaloutput[i][j]<<"$"<<std::endl;
        }
    }
}
int main() {
    clock_t start_time, end_time;
    start_time = clock();
    Decimator *decimator = new Decimator();
    decimator->initialize(8820.0, 4096.0, 5);

    FILE *fp = fopen("test_data.txt", "r");
    if (fp == NULL)
    {
        printf("error");
    }
    int fs = 8820;//had decimated frequency sampling rate by_YJY

    float data[44100 * 3] = { 0 };
    float output[(44100 * 3) / 5] = { 0 };

    for (int i = 0; i < 44100 * 3; i++)
        fscanf(fp, "%f", &data[i]);
    /*
       for(int i=0; i<4410; i++){
       printf("%f ", data[i]);
       if((i+1)%10==0) printf("\n");
       }
       std::cout<<"data size : "<<sizeof(data)/sizeof(data[0])<<std::endl;*/ 
    
    decimator->decimate(data, output, (44100 * 3) / 5);

    FILE *fp2 = fopen("deci.txt", "w+");
    for (int i = 0; i<(44100 * 3) / 5; i++) {
    	fprintf(fp2,"%f ", output[i]);
    	//      if((i+1)%10==0) fprintf(fp2, "\n");
    }
     
    CArray output2((int)(44100*3*0.2));
    for (int i=0;i<output2.size();++i)
    {
    output2[i]=Complex(output[i]);
    }
    //int n = output2.size();
    //std::cout<<output2.size()<<std::endl;
    float preprocessed_signal[6][44100*3/5]={0};
    preprocessing(output2,preprocessed_signal);
    
    FILE *fp4 = fopen("preprocessedSignal.txt", "w+");
    for (int i = 0; i <6; i++) {
        for(int j=0;j<44100*3/5;++j)
        {
            fprintf(fp4,"%f ",preprocessed_signal[i][j]);
        }  
        fprintf(fp4,"\n");
    }
    fclose(fp4);
    /*
    for(int i=0;i<44100*3*0.2;++i)
    {
        std::cout<<"output2"<<output2[i]<<"output1"<<output[i]<<"\n";
    }*/
    
    //fft(output2);
    /*
    for(int i=0;i<output2.size();++i)
    {
        std::cout<<"output2"<<output2[i]<<"\n";
    }
    */
    //fclose(fp2);
    //S_YJY
    //Complex output2[(44100 * 3) / 5];
    //fft(output, output2, sizeof(output)/sizeof(output[0]));
    //dft(output, output2, (44100 * 3) / 5);
/*    FILE *fp3 = fopen("fftTEST.txt", "w+");
    for (int i = 0; i < (44100 * 3) / 5; i++) {
        if(output2[i].imag() >= 0)
           fprintf(fp3, "%f+%fi ", output2[i].real(),output2[i].imag());
        else
            fprintf(fp3, "%f%fi ", output2[i].real(),output2[i].imag());
        //      if((i+1)%10==0) fprintf(fp2, "\n");
    }
    fclose(fp3);*/
    //std::cout<<"sizeof"<<sizeof(output2)/sizeof(output2[0])<<"\n";
    fclose(fp2);
    //E_YJY

    fclose(fp);
    end_time = clock();
    printf("%f\n", ((float)(end_time - start_time)/CLOCKS_PER_SEC));
    return 0;
}
