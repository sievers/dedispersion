#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <assert.h>

//delay in seconds = DM0*dm/(freq^2) where freq is in MHz
#define DM0 4000.0


float **matrix(int n, int m)
{
  float *vec=(float *)malloc(sizeof(float)*n*m);
  float **mat=(float **)malloc(sizeof(float *)*n);
  memset(vec,0,n*m*sizeof(float));
  for (int i=0;i<n;i++)
    mat[i]=vec+i*m;
  return mat;
}
/*--------------------------------------------------------------------------------*/
float *vector(int n)
{
  float *vec=(float *)malloc(sizeof(float)*n);
  assert(vec);
  memset(vec,0,n*sizeof(float));
  return vec;
}
/*--------------------------------------------------------------------------------*/
int *ivector(int n)
{
  int *vec=(int *)malloc(sizeof(int)*n);
  assert(vec);
  memset(vec,0,n*sizeof(int));
  return vec;
}
/*--------------------------------------------------------------------------------*/

void dedisperse_kernel(float **in, float **out, int n, int m)
{
  int npair=n/2;
  for (int jj=0;jj<npair;jj++) {
    for (int i=0;i<m;i++)
      out[jj][i]=in[2*jj][i]+in[2*jj+1][i];
    for (int i=0;i<m-jj-2;i++) 
      out[jj+npair][i]=in[2*jj][i+jj]+in[2*jj+1][i+jj+1];
    
  }
}
/*--------------------------------------------------------------------------------*/
void dedisperse_kernel_2pass(float **in, float **out, int n, int m) 
{
  
  int nset=n/4;
  for (int jj=0;jj<nset;jj++) {
    //#pragma omp for
    for (int i=0;i<m-4*jj-4;i++) {
      out[jj][i]=in[4*jj][i]+in[4*jj+1][i]+in[4*jj+2][i]+in[4*jj+3][i];
      out[jj+nset][i]=in[4*jj][i+jj]+in[4*jj+1][i+jj]+in[4*jj+2][i+1+jj]+in[4*jj+3][i+1+jj];
      out[jj+2*nset][i]=in[4*jj][i+2*jj]+in[4*jj+1][i+2*jj+1]+in[4*jj+2][i+2*jj+1]+in[4*jj+3][i+2*jj+2];
      out[jj+3*nset][i]=in[4*jj][i+3*jj]+in[4*jj+1][i+3*jj+1]+in[4*jj+2][i+3*jj+2]+in[4*jj+3][i+3*jj+3];      
    }
  }
}
/*--------------------------------------------------------------------------------*/

int get_npass(int n)
{
  int nn=0;
  while (n>1) {
    nn++;
    n/=2;
  }
  return nn;
}
/*--------------------------------------------------------------------------------*/
void dedisperse(float **inin, float **outout, int nchan,int ndat)
{
  int npass=get_npass(nchan);
  printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  for (int i=0;i<npass;i++) {    
    //#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      dedisperse_kernel(in+j,out+j,bs,ndat);
    }
    bs/=2;
    float **tmp=in;
    in=out;
    out=tmp;
  }
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}
/*--------------------------------------------------------------------------------*/

void dedisperse_dual(float **inin, float **outout, int nchan,int ndat)
{
  int npass=get_npass(nchan);
  //printf("need %d passes.\n",npass);
  //npass=2;
  int bs=nchan;
  float **in=inin;
  float **out=outout;

  //the npasss-1 is so that we stop in time to hand the final pass to 
  //the single-step kernel in the event of an odd depth.
  for (int i=0;i<npass-1;i+=2) {    
#pragma omp parallel for
    for (int j=0;j<nchan;j+=bs) {
      dedisperse_kernel_2pass(in+j,out+j,bs,ndat);
    }
    bs/=4;
    float **tmp=in;
    in=out;
    out=tmp;
  }
  if (npass%2==1) {
    //do a single step if we come in with odd depth
    //printf("doing final step for odd depth with block size %d.\n",bs);
    for (int j=0;j<nchan;j+=bs)
      dedisperse_kernel(in+j,out+j,bs,ndat);
    float **tmp=in;
    in=out;
    out=tmp;
    
  }
  
  memcpy(out[0],in[0],nchan*ndat*sizeof(float));
  
}
/*--------------------------------------------------------------------------------*/
void dedisperse_2pass(float **dat, float **dat2, int nchan, int ndat)
{
  dedisperse_kernel(dat,dat2,nchan,ndat);
  dedisperse_kernel(dat2,dat,nchan/2,ndat);
  dedisperse_kernel(dat2+nchan/2,dat+nchan/2,nchan/2,ndat);
}

/*--------------------------------------------------------------------------------*/

void write_mat(float **dat, int n, int m, char *fname)
{
  FILE *outfile=fopen(fname,"w");
  fwrite(&n,sizeof(int),1,outfile);
  fwrite(&m,sizeof(int),1,outfile);
  fwrite(dat[0],sizeof(float),n*m,outfile);
  fclose(outfile);
}

/*--------------------------------------------------------------------------------*/
void find_peak(float **dat, int nchan, int ndat, int *best_chan, int *best_dat)
{
  float max=0;
  int ichan=0;
  int idat=0;
  for (int i=0;i<nchan;i++) 
    for (int j=0;j<ndat;j++) {
      if (dat[i][j]>max) {
	max=dat[i][j];
	ichan=i;
	idat=j;
      }
    }
  float slope=(float)ichan/(float)nchan;
  float flux=dat[ichan][idat]+dat[ichan][idat-1]+dat[ichan][idat+1];
  printf("I found a peak at slope %8.4f at time %d with average flux %8.4f\n",slope,idat,flux/(float)nchan);

  if (best_chan)
    *best_chan=ichan;
  if (best_dat)
    *best_dat=idat;

}
/*--------------------------------------------------------------------------------*/
int get_nchan_from_depth(int depth) 
{
  int nchan=1;
  for (int i=0;i<depth;i++)
    nchan*=2;
  return nchan;
}
/*--------------------------------------------------------------------------------*/
float *get_dm_channels(int depth, float nu1, float nu2)
{
  int nchan=get_nchan_from_depth(depth);
  float l1=1/nu2/nu2;
  float l2=1/nu1/nu1;
  
  float dl=(l2-l1)/(nchan-1);

  float *freqs=vector(nchan);
  for (int i=0;i<nchan;i++) 
    freqs[i]=1/sqrt(l1+(0.0+i)*dl);
  return freqs;
}
/*--------------------------------------------------------------------------------*/
float *get_chime_channels(int nchan, float nu1, float nu2)
{
  float *chans=vector(nchan);
  float dnu=(nu2-nu1)/nchan;
  for (int i=0;i<nchan;i++) 
    chans[i]=nu1+(i+0.5)*dnu;
  return chans;
}
/*--------------------------------------------------------------------------------*/
float get_diagonal_dm(float tsamp, float *freqs) {
  //diagonal DM is when the delay between adjacent channels
  //is equal to the sampling time
  //delay = dm0*dm/nu^2
  // delta delay = dm0*dm*(1//nu1^2 - 1/nu2^2) = dt
  float d1=1.0/freqs[0]/freqs[0];
  float d2=1.0/freqs[1]/freqs[1];
  float dm_max=tsamp/DM0/(d2-d1);
  return dm_max;
  
}
/*--------------------------------------------------------------------------------*/
int *remap_channels(float *cur_freq, int ncur, float *dm_freq, int nchan)
{
  //find the mapping between sets of channels so that first set is most closely mapped onto
  //the second set in lambda^2 
  int *chan_map=ivector(ncur);
  float *ll=vector(ncur);

  for (int i=0;i<ncur;i++)
    ll[i]=1/cur_freq[i]/cur_freq[i];

  float *lref=vector(nchan);
  for (int i=0;i<nchan;i++)
    lref[i]=1/dm_freq[i]/dm_freq[i];

  for (int i=0;i<ncur;i++) {
    float delt=fabs(ll[i]-lref[0]);
    chan_map[i]=0;
    for (int j=0;j<nchan;j++) {
      if (fabs(ll[i]-lref[j])<delt) {
	delt=fabs(ll[i]-lref[j]);
	chan_map[i]=j;
      }
    }
    
  }

  free(lref);
  free(ll);
  //now do a bit of error analysis 
  float max_err=0.0;
  float mean_err=0.0;
  float dm_ref=1000.0;
  for (int i=0;i<ncur;i++) {
    float lag1=DM0*dm_ref/cur_freq[i]/cur_freq[i];
    float lag2=DM0*dm_ref/dm_freq[chan_map[i]]/dm_freq[chan_map[i]];	
    float myerr=fabs(lag1-lag2);
    mean_err+=myerr;
    if (myerr>max_err)
      max_err=myerr;
  }
  mean_err/=ncur;
  printf("remapping channels gives a mean error of %12.5f and a max error of %12.5f seconds at a DM of %12.2f\n",mean_err,max_err,dm_ref);
  return chan_map;

}
/*--------------------------------------------------------------------------------*/
float **time_squish_data(float **dat, float *freqs, int nchan, int ndata, float tsamp, float max_dm)
{
  float **newdat=matrix(nchan,ndata/2);
  float mylag0=DM0*max_dm/(freqs[0]*freqs[0]);
  for (int i=0;i<nchan;i++) {
    float mylag=DM0*max_dm/(freqs[i]*freqs[i]);
    if (mylag<mylag0)
      mylag0=mylag;
  }
  for (int i=0;i<nchan;i++) {
    float mylag=DM0*max_dm/(freqs[i]*freqs[i]);
    int dl=(mylag-mylag0)/tsamp;
    for (int j=0;j<ndata-dl-2;j+=2)
      newdat[i][j/2]=dat[i][j+dl]+dat[i][j+dl+1];
    
  }
  return newdat;
}
/*--------------------------------------------------------------------------------*/
float **dedisperse_chime(int ndata,  int chime_nchan, float *chime_freqs, int depth, float tsamp, float *dm_max, float **dat)
{
  //do a disperson transform on chime-type data. depth sets the number of output channels
  //in the transform.  tsamp sets the max dispersion measure to which we are sensitive
  //routine will return a 2-d array that has the dedispersed data transform, which
  //will be 2^depth by ndata in size.  

  int nchan=get_nchan_from_depth(depth);
  float *freqs=get_dm_channels(depth,chime_freqs[0],chime_freqs[chime_nchan-1]);
  *dm_max=get_diagonal_dm(tsamp,freqs);
  
  printf("limiting freqs are %12.4f %12.4f, from %12.4f %12.4f\n",freqs[0],freqs[nchan-1],chime_freqs[0],chime_freqs[chime_nchan-1]);

  float **dat1=matrix(nchan,ndata);
  float **dat2=matrix(nchan,ndata);
  int *chan_map=remap_channels(chime_freqs,chime_nchan,freqs,nchan);

  double t1=omp_get_wtime();

  for (int i=0;i<chime_nchan;i++)
    for (int j=0;j<ndata;j++)
      dat1[chan_map[i]][j]+=dat[i][j];

  printf("remapped data in %12.7f on %d %d\n",omp_get_wtime()-t1,chime_nchan,ndata);

  //dedisperse(dat1,dat2,nchan,ndata);
  dedisperse_dual(dat1,dat2,nchan,ndata);
  printf("dedispersed data in %12.7f on %d %d\n",omp_get_wtime()-t1,chime_nchan,ndata);

  free(dat2[0]);
  free(dat2);
  return dat1;

}
/*--------------------------------------------------------------------------------*/
float **simple_init_chime(int ndata, float *freqs, int nchan, float tsamp)
{
  float **data=matrix(nchan,ndata);
  float dm=304.6;
  for (int i=0;i<nchan;i++) {
    float delay=dm*DM0/(freqs[i]*freqs[i]);
    int isamp=delay/tsamp+50;
    if (i==nchan-1)
      printf("burst starts in data at sample %d\n",isamp);
	
    data[i][isamp]=1.0;
  }
  return data;
}

/*================================================================================*/

int main(int argc, char *argv[])
{


  int ndat=48000;
  if (argc>1)
    ndat=atoi(argv[1]);



  //set the number of output channels - we will have 2^depth output channels
  int depth=14;
  int nchan=get_nchan_from_depth(depth);

  int nchan_chime=4096;
  float *chime_freqs=get_chime_channels(nchan_chime,400,800);
  float tsamp=1e-3;
  float dm_max=0,dm_max2=0;

  float **dat=simple_init_chime(ndat,chime_freqs,nchan_chime,tsamp);
  
  
  float **dedispersed,**dat_shift,**dedispersed2;
  int niter=1;
  for (int i=0;i<niter;i++) {
    double t1=omp_get_wtime();
    dedispersed=dedisperse_chime(ndat,nchan_chime,chime_freqs,depth,tsamp,&dm_max,dat);
    dat_shift=time_squish_data(dat,chime_freqs,nchan_chime,ndat,tsamp,dm_max);
    float tot1=0,tot2=0;
    for (int ii=0;ii<nchan_chime;ii++) 
      for (int j=0;j<ndat;j++)
	tot1+=dat[ii][j];
    for (int ii=0;ii<nchan_chime;ii++)
      for (int j=0;j<ndat/2;j++)
	tot2+=dat_shift[ii][j];
    printf("tots are %12.4f %12.4f\n",tot1,tot2);

    dedispersed2=dedisperse_chime(ndat/2,nchan_chime,chime_freqs,depth,tsamp*2,&dm_max2,dat_shift);
 
    double t2=omp_get_wtime();
    printf("did 2-pass depth search to DM=%12.4f in %12.4f seconds.\n",dm_max2+dm_max,t2-t1);

    if (i<niter-1) {
      free(dedispersed[0]);
      free(dedispersed);
      free(dat_shift[0]);
      free(dat_shift);
      free(dedispersed2[0]);
      free(dedispersed2);
    }
  }
  printf("max dm to which we are sensitive is %12.4f\n",dm_max);

  int best_chan, best_dat;
  find_peak(dedispersed,nchan,ndat,&best_chan, &best_dat);
  printf("peak is at %d %d with %12.4f\n",best_chan,best_dat,dedispersed[best_chan][best_dat]);
  float tot1=0;
  float tot2=0;
  for (int i=best_dat-1;i<=best_dat+1;i++)
    tot1+=dedispersed[best_chan][i];
  for (int i=best_dat-2;i<=best_dat+2;i++)
    tot2+=dedispersed[best_chan][i];
  float best_dm=dm_max*best_chan/(0.0+nchan);
  printf("fluxes are %12.4f %12.4f at DM %12.4f\n",tot1,tot2,best_dm);


  find_peak(dedispersed2,nchan,ndat/2,&best_chan, &best_dat);
  printf("peak is at %d %d with %12.4f\n",best_chan,best_dat,dedispersed2[best_chan][best_dat]);
  tot1=0;
  tot2=0;
  for (int i=best_dat-1;i<=best_dat+1;i++)
    tot1+=dedispersed2[best_chan][i];
  for (int i=best_dat-3;i<=best_dat+3;i++)
    tot2+=dedispersed2[best_chan][i];
  float column_tot=0;
  for (int i=0;i<ndat/2;i++)
    column_tot+=dedispersed2[best_chan][i];
  float best_dm2=dm_max+dm_max2*best_chan/(0.0+nchan);
  printf("fluxes are %12.4f %12.4f %12.4f at DM %12.4f\n",tot1,tot2,column_tot,best_dm2);
}
