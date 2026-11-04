# Upstream provenance

Repository: [Azure/azureml-examples](https://github.com/Azure/azureml-examples)

Pinned commit: `27f4487a6060b80982ef0793e6ebcea90f509d37`

License: [MIT](LICENSE), retained verbatim from that commit.

The five source files below are unmodified originals. Their paths under
`upstream/` match their paths in the upstream repository:

| Path | Purpose |
|---|---|
| `sdk/python/jobs/single-step/lightgbm/iris/src/main.py` | LightGBM/Iris training source |
| `sdk/python/jobs/pipelines/1a_pipeline_with_components_from_yaml/train_model.yml` | Training component definition |
| `sdk/python/jobs/pipelines/1a_pipeline_with_components_from_yaml/train_src/train.py` | Training component source |
| `sdk/python/jobs/pipelines/1a_pipeline_with_components_from_yaml/score_data.yml` | Scoring component definition |
| `sdk/python/jobs/pipelines/1a_pipeline_with_components_from_yaml/score_src/score.py` | Scoring component source |

For the complete originals, including notebooks, datasets, environments, and
additional components, use the pinned upstream
[Iris example](https://github.com/Azure/azureml-examples/tree/27f4487a6060b80982ef0793e6ebcea90f509d37/sdk/python/jobs/single-step/lightgbm/iris)
and [component pipeline example](https://github.com/Azure/azureml-examples/tree/27f4487a6060b80982ef0793e6ebcea90f509d37/sdk/python/jobs/pipelines/1a_pipeline_with_components_from_yaml).
This selected subset is not a complete runnable upstream example and is not
preconfigured for KubernetesCompute. Review upstream notebooks before executing:
they can submit jobs, perform sweeps, or configure resources outside this sample's scope.

The root submission/operation helpers and all files in `samples/` are separately
authored examples. They are not Microsoft-published samples or upstream
qualification evidence.
