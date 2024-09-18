from setuptools import setup

setup(
    name='nudge-ft',
    version='0.1.0',    
    description='NUDGE: Lightweight Non-Parametric Embedding Fine-Tuning',
    url='https://github.com/szeighami/nudge',
    author='Sepanta Zeighami',
    author_email='zeighami@berkeley.edu',
    license='MIT',
    packages=['nudge'],
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)
