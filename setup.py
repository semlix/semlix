#!python

import os.path, sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.abspath("src"))
from semlix import versionstring


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        pytest.main(self.test_args)


if __name__ == "__main__":
    setup(
        name="semlix",
        version=versionstring(),
        package_dir={'': 'src'},
        packages=find_packages("src"),

        author="Alberto Ferrer",
        author_email="albertof@barrahome.org",

        description="Fast, pure-Python full text indexing, search, and spell checking library with semantic search capabilities.",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",

        license="Two-clause BSD license",
        keywords="index search text spell semantic",
        url="https://github.com/semlix/semlix",

        zip_safe=True,
        install_requires=['cached-property'],
        tests_require=['pytest'],
        extras_require={
            'semantic': [
                'numpy',  # Required for semantic search
            ],
            'semantic-full': [
                'numpy',
                'sentence-transformers',  # For SentenceTransformerProvider
                'openai',  # For OpenAIProvider
                'cohere',  # For CohereProvider
                'huggingface_hub',  # For HuggingFaceInferenceProvider
                'faiss-cpu',  # For FaissVectorStore (or faiss-gpu for GPU)
            ],
        },
        cmdclass={'test': PyTest},

        classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
        ],
    )
