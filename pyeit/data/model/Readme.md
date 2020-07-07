# Mesh Models

  - [x] `DLS2.mes` human head annotated mesh
  - [x] `I0007.mes` small scaled down human head mesh
  - [x] `lung.mes` human throax annotated mesh

These binary mesh models are installed by `setup.py` by default, and can be used using

```python
import pkg_resources
mstr = pkg_resources.resource_filename('pyeit', 'model/I0007.mes')
```

where `mstr` is the path to the mesh model.

