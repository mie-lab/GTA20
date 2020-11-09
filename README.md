[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mie-lab/GDA20/master)

# Geoinformationstechnologien und –analysen HS2020: Übungen

In diesem Github Repository finden Sie die Unterlagen zu den Übungen 10-12 der Vorlesung Geoinformationstechnologien und -analysen, durchgeführt im Herbstsemester 2020.

## Übung 10: Spatial Data Mining

In dieser Übung geht es darum, mittels Clustering-Methoden Stoppunkte in GPS-Trajektorien zu erkennen.

[Aufgabe](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=10_spatial_data_mining%2F10_spatial_data_mining_main.ipynb) / [Lösung](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=10_spatial_data_mining%2F10_spatial_data_mining_solution.ipynb)

## Übung 11: Analyse von Bewegungsdaten 1

Basierend auf Übung 10 werden hier die Bewegungsdaten anhand der Stoppunkte aufgeteilt.
Die resultierenden "triplegs" können dann auf verschiedene Arten weitergehend analysiert werden.

[Aufgabe](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=11_analyse_von_bewegungsdaten_1%2F11_analyse_von_bewegungsdaten_1_main.ipynb) / [Lösung](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=11_analyse_von_bewegungsdaten_1%2F11_analyse_von_bewegungsdaten_1_solution.ipynb)

## Übung 12: Analyse von Bewegungsdaten 2

Schlussendlich werden weitere Kontextdaten verwendet, um in Kombination mit den Bewegungsdaten aussagen über die Transportmittel oder den exakten Weg zu machen.

[Aufgabe](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=12_analyse_von_bewegungsdaten_2%2F12_analyse_von_bewegungsdaten_2_main.ipynb) / [Lösung](https://mybinder.org/v2/gh/mie-lab/GTA20/master?filepath=12_analyse_von_bewegungsdaten_2%2F12_analyse_von_bewegungsdaten_2_solution.ipynb)

## Wichtiges

Die Installation von `pyrosm` hat lange nicht funktioniert. 
Der Trick ist, `pyrosm` zu klonen, im `setup.py` die `setup_requirements` auf `setuptools` zu beschränken und dann alles mit `pip install -e .` zu installieren. 
Auf Linux (myBinder) geht das hoffentlich einfacher.