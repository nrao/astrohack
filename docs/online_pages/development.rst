Development
~~~~~~~~~~~

Setting up a development environment
####################################

Create a fresh environment, clone AstroHACK and make an editable
install (see :doc:`astrohack_installation`):

.. code-block:: sh

   $ conda create --name astrohack python=3.13 --no-default-packages
   $ conda activate astrohack
   $ git clone https://github.com/nrao/astrohack.git
   $ cd astrohack
   $ pip install -e .

pre-commit
##########

Install `pre-commit <https://pre-commit.com/>`_ and its git hooks:

.. code-block:: sh

   $ pip install pre-commit
   $ pre-commit install

Every ``git commit`` then runs the checks configured in
`.pre-commit-config.yaml <https://github.com/nrao/astrohack/blob/main/.pre-commit-config.yaml>`_:
`Ruff <https://docs.astral.sh/ruff/>`_ (linting, import sorting and
formatting, configured in ``ruff.toml``), ``pyupgrade``,
``absolufy-imports`` and file hygiene checks. The same checks run in a
GitHub Action on every push and pull request. If a hook rewrites files,
the commit is aborted: stage the rewritten files (``git add``) and
commit again. To run all checks on the whole repository:

.. code-block:: sh

   $ pre-commit run --all-files

Running the tests
#################

The tests download their data from the ToolVIPER data registry, so they
need network access:

.. code-block:: sh

   $ python -m pytest tests/unit

Submitting code
###############

Open an issue on `GitHub <https://github.com/nrao/astrohack/issues>`_,
create a branch for it from ``main`` and open a pull request against
``main``.
