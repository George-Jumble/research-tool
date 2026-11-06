#include "src/include.h"
/* Superblock driver: sb_bbn.x eta dNnu tau_n tablefile|none failsafe
   table file: lines "T_GeV rho_GeV4", any order (sorted here to descending T) */
int main(int argc,char**argv){
  struct relicparam p; double r[NNUC+1];
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
  p.err=0; nucl(&p,r);
  printf("%.6e %.6e %.6e %.6e %.6e\n",r[6],r[3],r[5],r[8],r[9]);
  return 0;}
