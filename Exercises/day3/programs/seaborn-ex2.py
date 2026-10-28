import seaborn as sb
from matplotlib import pyplot as plt
### May need to uncomment next 2 lines if running directly from cmd line:
#import matplotlib
#matplotlib.use("TkAgg")

mpg = sb.load_dataset('mpg')
sb.clustermap(mpg.corr(numeric_only=True), annot=True, fmt=".2f",
              cbar_kws={'label':'Correlation Coefficients'})
### Use next line if copying to IDE or running directly at cmd line
plt.show()
### comment line above, uncomment line below if running from batch script
#plt.savefig('mpg_clustermap',format='pdf')
### title and format file as desired
