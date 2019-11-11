library(ggplot2)
library(ggrepel)

df = read.csv("C:\\Users\\j\\Documents\\CHau\\hw7\\pca.csv", header = TRUE, sep=",")

pca1 = prcomp(df, scale = TRUE)

circle = function(center = c(0, 0), npoints = 100) 
{
  r = 1
  tt = seq(0, 2 * pi, length = npoints)
  xx = center[1] + r * cos(tt)
  yy = center[1] + r * sin(tt)
  return(data.frame(x = xx, y = yy))
}

corcir = circle(c(0, 0), npoints = 100)
correlations = as.data.frame(cor(df, pca1$x))

arrows = data.frame(
  x1=c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
  y1=c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
  x2 = correlations$PC1, 
  y2 = correlations$PC2)

ggplot() + 
  geom_path(
    data = corcir, aes(x = x, y = y), 
    colour = "gray65") + 
  geom_segment(data = arrows, 
               aes(x = x1, y = y1, xend = x2, yend = y2), 
               colour = "gray65") + 
  geom_text_repel(data = correlations, 
            aes(x = PC1, y = PC2, 
                label = rownames(correlations))) + 
  geom_hline(yintercept = 0, colour = "gray65") +
  geom_vline(xintercept = 0, colour = "gray65") +
  xlim(-1.1, 1.1) + ylim(-1.1, 1.1) + 
  labs(x = "pc1 axis", y = "pc2 axis") + 
  ggtitle("Circle of PCs")