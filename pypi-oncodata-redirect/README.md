# oncodata has been renamed to **oncoref**

This package is no longer maintained under the name `oncodata`. It has been
renamed to [`oncoref`](https://pypi.org/project/oncoref/).

## Migrate

```bash
pip uninstall oncodata
pip install oncoref
```

```python
import oncoref          # was: import oncodata
```

The `oncodata` command remains as a deprecated alias for `oncoref`.

This final `oncodata` release depends on `oncoref`, re-exports its public API,
and preserves the legacy module paths and optional `plots` extra. It will not
receive feature updates. Please switch to `oncoref`.

Source & issues: <https://github.com/pirl-unc/oncoref>
