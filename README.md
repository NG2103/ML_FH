# allgemein

* code aufrämen und diskussion fehlen natürlich noch überall ;-) 

# linear model

* `plt.title("Nested cross test predictions on noisy sin-wave")`-> hier habe ich es nicht geschafft Linien zu zeichnen
* Vergleich R2 und mean squared error -> Diskussion fehlt
* Application to nonlinear prediction problems
	* ich habe Misra1a und Gauss1 probiert mit KRR finde aber keine sinnvollen alpha und gamma Werte -> die Vorhersage für Gauss1 ist nicht so schlecht, aber für Misra1a funktioniert es gar nicht
	* neben KRR können wir auch MLPRegressor oder GPR verwenden, wobei er gesagt hat, dass GPR sehr rechenintensiv und deshlab MLPRegressor

# clustering

* hier fehlt eigentlich nur mehr das von ihm genannten und OPTIONALE spectral clustering
* https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
* sonst ist fertig bis auf die Diskussion

# Classification

* da fehlt eigentlich noch (fast) alles
* ich habe mal versucht die grid search in KNN zu implementieren aber ich denke da passt noch was nicht ;-)


# Updates
## Daniel 30.10. & 1.11.

### linear model
- parameter für linspace geändert, weil 0 eine Warnung ausgibt und ewig rechnet weil es auf einen anderen solver wechselt
- Fortschrittsbalken eingefügt --> bitte über conda tqdm installieren
- Plotten als Linien funktioniert jetzt --> xnl sind zufallszahlen --> nicht der Größe nach sortiert

- Beim NIST waren teilweise noch die Daten von der Sinuswelle drin, deswegen hat sich das ganze komisch verhalten
- NIST KRR ist jetzt mit drei verschiedenen Scores implementiert
- Hab auch mal MLPRegressor probiert, funktioniert aber momentan noch gar nicht