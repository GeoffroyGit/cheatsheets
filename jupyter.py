# put this at the beginning of the jupyter notebook to enable autoreload

%load_ext autoreload
%autoreload 2

# visualizing pipelines in HTML

from sklearn import set_config
set_config(display='diagram')
