## Real data

2 real datasets are used in experiments:

- household eletricity consumption by 3 entities: appliance in kitchen, in loundry room, from air heater / conditioner. Therefore we have three 1d signals. [Link](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- accelerometry measurement's of human if 3d space, so we also have three 1d signals here. [Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)


## Demo

Here we test first version of tSSA algorithm on 3 datasets: synthetical (3 sine's with diffrent frequences, electricity consumption, accelerometry). Demo contains data extraction code parts which are not really interesting. The method itself consist in stacking trajectory matrices of$$ 1d signals, forming tensor. After, we decompose it using CP tensor decomposition with rank = $ r $, getting for each i-th signal:

$$
    TrajTensor_i = \sum\limits_{j = 1}^r c_j[i] \cdot a_j \otimes b_j  
$$

We can see that all trajectory matrices are decomposed by the same factors but with diffrent 'singular' values. Getting it, method groups **tensor** factors (namely c_j \otimes a_j \otimes b_j), then sum up factors within each group, obtaining new factor for each group. Each factor's slice is grouped trajectory matrices for each signal. Hankelizing them and extracting component signals for each initial signal is the final step. Grouping here is handled based on the norms of $ c_j $ vectors as some weight of each factor.

Results of decomposing each signal as well as result of restorement of each signal are demonstraited.


## Model_comparasion

Here are realized 2 diffrent methods to decompose multidimensional sigmnals.

One is 'plain' mSSA which stacks trajectory matrices for each signal in one matrix horizontaly and then performaing classicalSSA on this. After decomposition, gained factors are destacked to obtain classical SSA decomposition for each signal. Then theyare grouped (for each signal individually), hankelized and component-signals retreived. The results of decomposition for eachsignal and restoration are displayed. Data here - electricity.

Second method here is similar to one in Demo, but here we do not group whole tensor factors. After CP decomposition we obtain decomposition of traj. matrix for each signal (like in formula in Demo) and then group, hankelize and extract components for each signal individually. Grouping are now based on $ c_j[i], j \in {1, \ldot, r} $. This is more flexible method and grouping now is more intuitive.


## Demo_1

Another tSSA method. Here we form trajectory tensor diffrently: having time grid (days, months) and signal's value for each point in grid we build such matrix for each signal, then stack it into tensor. The rest part is similar to previous method: decompose tensor, obtain decomposition for each individual signal, group. The hankelization here is absent because intial trajectory tensors do not possess any hankel structure. Decompositions and restorements are displayed.
