---
title: 'GNN Spatial Exploration'
collection: blogs
date: 2020-11-30
permalink: /posts/2012/11/gnn-spatial-exploration/
---

### Brief exploration: Graph Neural Networks and Spatial Statistics

I was thinking about spatial statistics and was curious about the place Graph Neural Networks may have.  At a high level, spatial statistics has methods for leveraging local information to improve a prediction.  I wondered how Graph Neural Networks would perform if we used a graph to provide relational information about locations.

I decided to explore this as a fun side-project.  This is undoubtedly a project worthy of careful research and thinking but I decided to try whatever felt natural without getting bogged down in too many technical details.  Surely this is a great way to learn a lot in a short period of time (spatial statistics is not an area I know much about!).  

# Overview

## Analysis

Suppose we want to predict the rainfall in various locations.  A naive approach might be to collect features (independent variables) at each region, build a linear model, and use OLS regression to estimate the model.  However, it would be arguably better to also leverage spatial information so that the response function behaves more similarly for locations that are "close".  Local regression techniques exist for this purpose.  (Reference: [Hijmans and Ghosh, Chapter 6](https://rspatial.org/raster/analysis/analysis.pdf).

What if we built a graph that joined nearby locations and used a Graph Neural Network?  How would an off-the-shelf Graph Neural Network compare against the "global" linear regression model, without too much tuning? 

## Code

We will use the California precipitation data which is already available using `R` software.  Thus, we pre-process the data in `R` before using PyTorch Geometric and scikit-learn for analysis.  The code is available as a repo on my [GitHub](https://github.com/rmurphy2718/gnn-spatial-exploration) and the software dependencies are pretty straightforward to download using internet resources.





# Obtain and pre-process the data in R

The California precipitation data is available in R’s `spatial` package.
The data contains monthly precipitation, altitude, and geocodes
(latitude/longitude) for 456 locations.

``` r
library("rspatial")
```

``` r
p <- sp_data("precipitation")
print(dim(p))
```

    ## [1] 456  17

``` r
print(names(p))
```

    ##  [1] "ID"   "NAME" "LAT"  "LONG" "ALT"  "JAN"  "FEB"  "MAR"  "APR"  "MAY" 
    ## [11] "JUN"  "JUL"  "AUG"  "SEP"  "OCT"  "NOV"  "DEC"

``` r
print(head(p[, c("NAME", "LAT", "LONG", "ALT", "JAN", "DEC")]))
```

    ##                   NAME   LAT    LONG ALT  JAN DEC
    ## 1         DEATH VALLEY 36.47 -116.87 -59  7.4 3.9
    ## 2  THERMAL/FAA AIRPORT 33.63 -116.17 -34  9.2 5.5
    ## 3          BRAWLEY 2SW 32.96 -115.55 -31 11.3 9.7
    ## 4 IMPERIAL/FAA AIRPORT 32.83 -115.57 -18 10.6 7.3
    ## 5               NILAND 33.28 -115.51 -18  9.0 9.0
    ## 6        EL CENTRO/NAF 32.82 -115.67 -13  9.8 1.4

In Section 6.1 of [Spatial Data Analysis with
R](https://rspatial.org/raster/analysis/analysis.pdf) the authors
predict the annual precipitation from altitude and relational spatial
information. Following their analysis, we sum across the 12 months to
create the target variable. Additionally, to pre-process for deep
learning, we scale our only non-relational feature, altitude, and
shuffle the rows.

``` r
p$pan <-rowSums(p[,6:17])  # Columns 6:17 are Jan - Dec
p$scaled.alt <- scale(p$ALT)  # "0-mean 1-std" standardization for simplicity

# Shuffle dataset
set.seed(42)
n <- nrow(p)
p <- p[sample(n), ]
```

## Exploiting spatial information by building a graph

In the textbook mentioned above, the authors use local regression to
exploit spatial information when predicting rainfall. I thought it would
be fun to try to leverage spatial relational information with a graph
neural network.

So, we will build a graph where the locations are vertices and edges are
formed between nearest locations. Our first step, then, is to build a
pairwise distance matrix that stores the distance between every pair of
locations.

I found a function that takes in a pair of latitudes/longitudes and
returns distances from the blog at
[exploratory.io](https://exploratory.io/).

``` r
# We need the following helper function from exploratory.io,
# which we can simply copy-and-paste or obtain by installing their software.
# (I didn't feel like updating my dependencies etc. for one function)
# https://rdrr.io/github/exploratory-io/exploratory_func/src/R/util.R#sym-list_extract
list_extract <- function(column, position = 1, rownum = 1){
  
  if(position==0){
    stop("position 0 is not supported")
  }
  
  if(is.data.frame(column[[1]])){
    if(position<0){
      sapply(column, function(column){
        index <- ncol(column) + position + 1
        if(is.null(column[rownum, index]) | index <= 0){
          # column[rownum, position] still returns data frame if it's minus, so position < 0 should be caught here
          NA
        } else {
          column[rownum, index][[1]]
        }
      })
    } else {
      sapply(column, function(column){
        if(is.null(column[rownum, position])){
          NA
        } else {
          column[rownum, position][[1]]
        }
      })
    }
  } else {
    if(position<0){
      sapply(column, function(column){
        index <- length(column) + position + 1
        if(index <= 0){
          # column[rownum, position] still returns data frame if it's minus, so position < 0 should be caught here
          NA
        } else {
          column[index]
        }
      })
    } else {
      sapply(column, function(column){
        column[position]
      })
    }
  }
}

# Here is the function that computes distance.
# https://blog.exploratory.io/calculating-distances-between-two-geo-coded-locations-358e65fcafae
get_geo_distance = function(long1, lat1, long2, lat2, units = "miles") {
  loadNamespace("purrr")
  loadNamespace("geosphere")
  longlat1 = purrr::map2(long1, lat1, function(x,y) c(x,y))
  longlat2 = purrr::map2(long2, lat2, function(x,y) c(x,y))
  distance_list = purrr::map2(longlat1, longlat2, function(x,y) geosphere::distHaversine(x, y))
  distance_m = list_extract(distance_list, position = 1)
  if (units == "km") {
    distance = distance_m / 1000.0;
  }
  else if (units == "miles") {
    distance = distance_m / 1609.344
  }
  else {
    distance = distance_m
    # This will return in meter as same way as distHaversine function. 
  }
  distance
}
```

Now, to compute the distance matrix, I loop over all pairs of locations
and apply this function. I could not immediately think of a solution
that would be cleaner than looping in this case, but I welcome ideas
from readers\!

``` r
# Create distance matrix (this takes a few moments, unsurprisingly)
distances <- matrix(nrow = n, ncol = n, data = -1)
colnames(distances) <- p$NAME
rownames(distances) <- p$NAME

for(ii in 1:n){
  for(jj in 1:ii){  # Can't run jj to (ii-1). For ii==1, this would yield sequence (1, 0).
    if(ii == jj){
      distances[ii, ii] <- 0.0
    }else{
      distances[ii, jj] <- get_geo_distance(long1=p$LONG[ii], lat1=p$LAT[ii],
                                            long2=p$LONG[jj], lat2=p$LAT[jj],
                                            units='km')
      distances[jj, ii] <- distances[ii, jj]
    }
  }
}
```

A summary of the distance matrix:

``` r
summary(as.numeric(distances))  # The min should be 0, not -1.
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     0.0   203.5   369.1   418.8   607.1  1273.9

**Double check** It’s always good to double-check the data. I found an
online tool that computes the distance “as the crow flies” between
locations and spot-checked two pairs. Both checks match within a few KM.
We don’t expect an exact match since many pairs of latitude/longitude
can originate from the same city, and it may not be the case that the
web tool uses exactly the same geocodes as in the precipitation data.

``` r
# Compare with https://www.freemaptools.com/how-far-is-it-between.htm
print(distances['PARADISE', 'SANTA CLARA         USA'])
```

    ## [1] 262.1227

``` r
print(distances['BIG SUR             USA', 'LONG BEACH, CA   11'])
```

    ## [1] 432.1773

Equipped with the distance matrix, we can use the `nng` (nearest
neighbor graphs) function from `cccd`. The value of \(k\), the number of
nearest neighbors is undoubtedly an important hyperparameter, but I
simply chose 5 for now. I will update if I come back to this.

``` r
library("cccd")
library("igraph")
```

``` r
g <- as.undirected(
  nng(dx=distances, k=5, mutual=FALSE)   # (mutual=FALSE then undirected) != using mutual=TRUE)
)

# Another sanity-check:
neighbor.idx <- neighbors(g, 1, "all")  # indices of locations nearest to 1, per the graph
closest.idx <- order(distances[1, ])[2:6]  # indices of locations nearest to 1, per distance mat
all(sort(neighbor.idx) == sort(closest.idx))  # compare after sorting (igraph returns arbitrary order)
```

    ## [1] TRUE

Note, `nng` does not force all vertices to have exactly 5 edges, so it
is not a regular graph.

## Exporting the data

Finally, we save these to disk for processing in
python/PyTorch-Geometric.

``` r
# Write to "raw" directory, per convention of PyTorch-Geomtric.
write.csv(p$pan, file='raw/targets.csv')
write.csv(p$scaled.alt, file='raw/features.csv')

# Save graph as a graphml so it can be loaded by networkx.
write_graph(g, file = 'raw/ca_distance_graph.graphml', format = 'graphml')
```

# Modeling with a Graph Neural Network

Now, we pose this as a vertex regression task.  That is, we have a graph $G = (V, E)$ with associated vertices $V = \{1, 2, \ldots, 456 \}$ representing locations and edges $E$ representing nearest neighbor relationships.  Each vertex $v \in V$ is also associated with an input feature $x_v \in \mathbb{R}$ representing the (scaled) altitude as well as a target value $y_v \in \mathbb{R}$ representing the precipitation.  We will use a [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) to predict the $y_v$ values.

## Loading the data

We will load the data pre-processed in R into a Pytorch Geometric InMemoryDataset using the [convert_pytorch_geometric](https://github.com/rmurphy2718/gnn-spatial-exploration/blob/main/convert_pytorch_geometric.py) script.  


```python
import torch
import torch_geometric
import torch_geometric.transforms as transforms
from convert_pytorch_geometric import CaRain, create_pyg

torch.manual_seed(42)  # Set a seed for consistency in blogging.
```




    <torch._C.Generator at 0x7fb2fc064750>




```python
dataset = CaRain(".", pre_transform=create_pyg)
g = dataset[0]

# Note: nearest neighbors graph is not regular; that is, not all vertices have the same degree.
# So, we can add a one-hot encoding of the vertex degree as an additional feature as is sometimes done with featureless graphs, 
# in case it carries extra useful information.
degrees = torch_geometric.utils.degree(g.edge_index[0])
max_deg = torch.max(degrees).long().item()
add_degree_fn = transforms.OneHotDegree(max_deg)
add_degree_fn(g)

```




    Data(edge_index=[2, 2772], x=[456, 13], y=[456])



Next, we partition the vertex set into train and test sets.  We will leave it at that, although using multiple random sets and model initializations would provide a better assessment of generalization performance.  I will update this blog if I make this enhancement.

Specifically, we create boolean indicators -- masks -- that indicate whether a vertex belongs to a given set. 


```python
def make_train_test_masks(data_size, train_size):
    # Creat a boolean with exactly `train_size` are True, rest False
    unshuffled_train = int(train_size) > torch.arange(data_size)
    
    # shuffle to get train mask
    train_mask = unshuffled_train[torch.randperm(data_size)]
    
    # negate to get test mask
    test_mask = ~train_mask
    
    return train_mask, test_mask

g.train_mask, g.test_mask = make_train_test_masks(g.num_nodes, 0.2 * g.num_nodes)
```

## Train a GNN model

Next, we initialize an off-the-shelf Graph Convolutional Network given in the [PyTorch Geometric tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html).  I wonder whether this is enough to do better than the global regression?  In my experience, while refined Message Passing GNNs have been developed since GCN, it performs reasonably well and is a simple choice.


```python
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class Model(nn.Module):
    def __init__(self, in_dim):
        super(Model, self).__init__()
        self.conv1 = GCNConv(in_dim, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.25, training=self.training)  # In my experience, weaker dropout is good
        x = self.conv2(x, edge_index)

        return x.squeeze()
```

We train the model using the Adam optimizer for 5000 epochs.  Even on my local machine's CPUs, it only takes a few moments to train.


```python
model = Model(g.num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
crit = nn.MSELoss()

model.train()
for epo in range(5000):
    optimizer.zero_grad()
    pred = model(g)
        
    loss = crit(pred[g.train_mask], g.y[g.train_mask])
    loss.backward()
    optimizer.step()
    
    if epo % 500 == 0:
        print(loss.item())

```

    515239.75
    162703.046875
    150588.046875
    162187.421875
    152907.640625
    168840.875
    156450.796875
    153671.765625
    145880.078125
    145020.8125


Note that the MSE values are large due to the scale of the target.  Finally, we can evaluate the model on the vertices in the test set.


```python
model.eval()
test_pred = model(g)
test_loss = crit(pred[g.test_mask], g.y[g.test_mask]).item()
print(test_loss)
```

    184472.328125


# Compare with global linear regression

The [Spatial Data Anlysis textbook](https://rspatial.org/raster/analysis/analysis.pdf) uses (global) linear regression as the naive baseline.  Let us see how it performs.


```python
from sklearn.linear_model import LinearRegression
```


```python
# Convert to numpy
X = g.x[:, 0].unsqueeze(1).numpy()  # Remove degree information
y = g.y.numpy()
train_mask = g.train_mask.numpy()

# Train on training split (using altitude only)
lm = LinearRegression()
reg = lm.fit(X[train_mask], y[train_mask])

# Evaluate on test split
yhat = reg.predict(X[~train_mask])
crit(torch.from_numpy(yhat), g.y[g.test_mask]).item()
```




    186334.109375



# Results and discussion

Pulling from the results above, the Mean Squared Errors for the two models are shown below.

| GCN        | Global Linear Model |
|------------|---------------------|
| 184,472.33 | 186,334.11          |

While the error for GCN is smaller, it is likely not significant.  I am not showing error standard deviations across multiple runs here, which I may perform in the future.  For now, my curiosity was satisfies :).   Nonetheless, it is cool that GCN performs comparably out-of-the-box without tuning the number of neighbors or its model parameters. 

My curiosity was satisfied, but here are future steps I may take, and will update the blog accordingly.

## Next steps

* Explore the impact of $k$, the number of neighbors when building the graph.
* Use more thorough evaluation with multiple random splits.
* Explore different graph neural network models.
* Explore different feature engineering schemes other than appending the vertex degree.
