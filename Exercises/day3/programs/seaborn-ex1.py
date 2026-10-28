import seaborn as sb
### Uncomment next 2 lines if running directly from cmd line:
import matplotlib
matplotlib.use("TkAgg")

dat = sb.load_dataset('penguins')
g = sb.pairplot(data=dat, corner=True, hue='species')
### Use next 2 lines if running directly at cmd line
g.figure.show()
input("Press any key to exit")
### comment line above, uncomment line below if running from batch script
#g.figure.savefig('penguins_pairplot.png')
### title and format file as desired
