#include "src/include.h"
/* Superblock full-vector driver (same argv as sb_bbn.x):
   sb_bbn_full.x eta dNnu tau_n tablefile|none failsafe [xi] [Tnudec]
   prints: err NNUC ratioH[0..NNUC] Am[1..NNUC] Zm[1..NNUC]
   ratioH[0]=eta, ratioH[2]=Y_H (per baryon), ratioH[6]=4*Y_He4, others X/H;
   ratioH[5] includes H3, ratioH[8] includes Be7 (AlterBBN post-BBN folding). */
int main(int argc,char**argv){
  struct relicparam p; double r[NNUC+1];
  const double Am[27]={0.,1.,1.,2.,3.,3.,4.,6.,7.,7.,8.,8.,9.,10.,11.,11.,12.,12.,12.,13.,13.,14.,14.,14.,15.,15.,16.};
  const double Zm[27]={0.,0.,1.,1.,1.,2.,2.,3.,3.,4.,3.,5.,4.,5.,5.,6.,5.,6.,7.,6.,7.,6.,7.,8.,7.,8.,8.};
  double eta=6.1e-10,dN=0,tau=879.4; int fs=1; char*tf="none";
  if(argc>1) eta=atof(argv[1]); if(argc>2) dN=atof(argv[2]); if(argc>3) tau=atof(argv[3]);
  if(argc>4) tf=argv[4]; if(argc>5) fs=atoi(argv[5]); double xi=0., Tnd=-1.; if(argc>6) xi=atof(argv[6]); if(argc>7) Tnd=atof(argv[7]);
  Init_cosmomodel(&p); p.failsafe=fs;
  Init_cosmomodel_param(eta,p.Nnu,dN,tau,p.life_neutron_error,xi,0.,0.,&p); if(Tnd>0) p.Tnudec=Tnd;
  if(strcmp(tf,"none")){
    static double tab[2][NTABMAX]; int n=0; FILE*f=fopen(tf,"r");
    while(n<NTABMAX && fscanf(f,"%lf %lf",&tab[0][n],&tab[1][n])==2) n++; fclose(f);
    for(int i=0;i<n;i++)for(int j=i+1;j<n;j++) if(tab[0][j]>tab[0][i]){double a=tab[0][i],b=tab[1][i];tab[0][i]=tab[0][j];tab[1][i]=tab[1][j];tab[0][j]=a;tab[1][j]=b;}
    Init_dark_density_table(tab,n,&p);
  }
  p.err=0; int e=nucl(&p,r);
  printf("%d %d",e,NNUC);
  for(int i=0;i<=NNUC;i++) printf(" %.12e",r[i]);
  for(int i=1;i<=NNUC;i++) printf(" %g",Am[i]);
  for(int i=1;i<=NNUC;i++) printf(" %g",Zm[i]);
  printf("\n"); return 0;}
