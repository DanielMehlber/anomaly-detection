# Software Requirements Specification (SRS)
## System zur Erkennung von Sensor- und Bildanomalien in Langzeittests

---

## 1. Einleitung & Zielsetzung

### 1.1 Zweck des Dokuments
Dieses Dokument spezifiziert die funktionalen und nicht-funktionalen Anforderungen für ein minimalistisches, erweiterbares Software-Framework zur Erkennung von Bild- und Sensoranomalien. Es dient als eindeutige Implementierungsgrundlage (Baseline) für autonome Entwicklungs-Agenten sowie Software-Ingenieure. 

### 1.2 Problemstellung
Kamerasensoren werden in Klimakammern über mehrere Tage extremen Umgebungsbedingungen (Hitze, Kälte, Feuchtigkeit) ausgesetzt. Um Fehlerzustände wie Bildflackern, geometrische Verzerrungen oder den Ausfall von Kontrollelementen zu detektieren, mussten Prüfer bisher stundenlange Videos manuell sichten. Dieses System automatisiert diesen Prozess vollständig, lokal und ressourceneffizient.

### 1.3 Grundphilosophie: Eindeutigkeit vs. Flexibilität
* **Für den Agenten:** Die Kernarchitektur, die Datenschnittstellen und das Persistenzmodell sind präzise und unmissverständlich definiert.
* **Für das System:** Es darf **keine Hardcodierung** von Grenzwerten, Algorithmen-Parametern oder spezifischen Hardware-Eingabeformaten stattfinden. Das System muss als **generisches Framework** aufgebaut sein, dessen Verhalten über Konfigurationsdateien gesteuert wird und dessen Pipelines modular erweiterbar sind.

---

## 2. Systemarchitektur & Komponenten-Design

Das Framework basiert auf dem **Pipe-and-Filter-Architekturmuster**. Jede Analysekomponente arbeitet als isolierter Filter, der den Datenstrom liest, Metriken extrahiert und Anomalie-Events an einen zentralen Aggregator meldet.

### 2.1 Abstrakte Input-Schnittstelle (Erweiterbares Interface)
Das System darf nicht fest an ein bestimmtes Videoformat oder ein spezifisches Kamera-SDK gebunden sein.

* **Anforderung:** Definition einer abstrakten Basisklasse `AbstractInputProvider`.
* **Baseline-Implementierung:** Für die Demonstration wird die konkrete Klasse `VideoFileInputProvider` implementiert, welche lokale Videodateien (`.mp4`, `.avi`) einliest. Als einziger "Umweltfaktor" dient hierbei die relative Laufzeit (Zeitstempel) des Videos.
* **Erweiterbarkeits-Vorgabe:** Die Architektur muss so vorbereitet sein, dass zukünftige Implementierungen (z. B. `BaslerCameraInputProvider` über das Pylon-SDK oder `CombinedThermalInputProvider` über eine parallele CSV-Temperatur-API) per Plugin-Prinzip integriert werden können, ohne die nachgelagerte Analyse-Pipeline zu verändern.

### 2.2 Dynamische Initialisierung & Self-Calibration
Das System darf keine starren Schwellenwerte für "Normalzustände" voraussetzen, da sich Testaufbauten und Beleuchtungen unterscheiden.

* **Die Kalibrierungs-Minute:** Die ersten **60 Sekunden** des Datenstroms gelten als garantiert stabil, fehlerfrei und repräsentativ (statisches Kontrollbild/Leinwand).
* **Statistische Baseline:** In dieser Phase berechnet das System mathematische Basisdaten:
  * Mittlere Helligkeit und Standardabweichung des Pixelrauschens (zeitlich und räumlich).
  * Detektion und Registrierung von statischen Kontrollpunkten/Keypoints auf der Leinwand.
* **Toleranz-Generierung:** Das System leitet die initialen Akzeptanzgrenzen dynamisch aus dieser ersten Minute ab.

### 2.3 Modulare Analyse-Pipeline (Die Filter)
Die Filter sind voneinander isolierte Funktionseinheiten. Sie erhalten das Frame-Objekt (inklusive aktueller Umweltmetadaten) als Read-Only-Eingabe.

* **Vorbereinigungs-Stufe (Optional):** Ein optionaler, vorgeschalteter Filter darf globale Anpassungen vornehmen (z. B. globale Helligkeitsnormierung), um langsame, umweltbedingte Trends (z. B. Nachlassen einer Glühbirne über 48 Stunden) auszugleichen und False Positives zu minimieren.
* **Unabhängigkeit:** Tritt in Filter A ein Fehler auf (z. B. temporales Flackern), darf dies die algorithmische Ausführung von Filter B (z. B. spatiale Verzerrung) nicht blockieren. Algorithmische Interdependenzen sind deklarativ oder über nachgelagerte Logiken zu lösen, nicht durch Hardcodierung innerhalb der Filter.

---

## 3. Klassifizierung von Anomalien

Das System unterscheidet architektonisch strikt zwischen zwei Hauptklassen von Anomalien, für die jeweils ein erweiterbarer Basis-Filter zu implementieren ist.

| Anomalie-Klasse | Beschreibung | Baseline-Metriken (Beispiel) | Visualisierung im Report |
| :--- | :--- | :--- | :--- |
| **Zeitlich (Temporal)** | Anomalien, die sich erst durch den Vergleich aufeinanderfolgender Frames über die Zeitachse offenbaren. | • Flacker-Intensität<br>• Frequenz des Helligkeitssprungs<br>• Frame-Ausfallrate (Black Screen) | Repräsentativer Frame des Intensitäts-Peaks. |
| **Räumlich (Spatial)** | Anomalien, die innerhalb eines einzelnen Frames (strukturell oder geometrisch) auftreten. | • Grad der geometrischen Verzerrung (%)<br>• Anzahl vermisster Kontrollelemente (Keypoints) | Frame mit farblicher Hervorhebung (Bounding Box oder Maske) der Anomalie. |

---

## 4. Datenhaltung & Ereignis-Logik (Persistence)

Da die Testläufe über mehrere Tage auf Standard-Laptops laufen, ist eine **frame-basierte Speicherung (z. B. Abspeichern jedes fehlerhaften Bildes) strikt untersagt**, um Festplatten-Overhead und I/O-Engpässe zu vermeiden.

### 4.1 Das Event-Modell (Ereignisbasierte Aggregation)
Anomalien werden als kontinuierliche Zustände ("Events") erfasst:
1. **Event-Start:** Sobald ein Filter den dynamischen Toleranzbereich für eine konfigurierte Anzahl von Frames (Debounce-Time) überschreitet, wird ein Event erzeugt und ein Zeitstempel (`Start_Umweltfaktor`) gesetzt.
2. **Event-Update:** Solange die Anomalie anhält, aggregiert das System die statistischen Kennzahlen (z. B. Maximalwert der Intensität).
3. **Event-Ende:** Fällt der Wert wieder unter die Toleranzgrenze, wird das Event geschlossen und der End-Zeitstempel (`End_Umweltfaktor`) vermerkt.

### 4.2 Speicher-Infrastruktur
* **Datenbank:** Einsatz einer vollständig serverlosen, lokalen **SQLite-Datenbank**.
* **Daten-Minimalismus:** Pro Event wird genau **ein** repräsentatives Key-Frame (z. B. beim Peak der Anomalie) als komprimiertes Bild (JPEG) im Dateisystem gespeichert und der Pfad in der Datenbank hinterlegt. Das Bild enthält eine algorithmische Hervorhebung (Highlighting) der betroffenen Region.

---

## 5. Offline-Dashboard & Operator-Interface (UI)

Das Interface dient dem Prüfer zur Überwachung und historischen Analyse. Es muss **vollständig offline-fähig** sein.

### 5.1 Technische Vorgaben
* **Technologie:** Minimalistisches Python-Webframework (z. B. Streamlit, FastAPI mit nativem, einfachem HTML/CSS). Keine schweren externen JavaScript-Bibliotheken oder Cloud-Abhängigkeiten.
* **Kein technischer Overhead:** Der Verzicht auf komplexe Echtzeit-Protokolle (wie WebSockets) ist explizit vorgegeben. Eine einfache, konfigurierbare Client-seitige Aktualisierung (z. B. HTML-Meta-Refresh alle 5 Sekunden) reicht für die Live-Ansicht aus.

### 5.2 Funktionale Ansichten
1. **Live-Monitor:** Anzeige des aktuellen Status des laufenden Tests, der bisherigen Event-Anzahl und einer Kurzzusammenfassung.
2. **Historien-Archiv:** Ermöglicht das Laden und Sichten vergangener Testläufe aus der SQLite-Datenbank.
3. **Event-Report:** Chronologische und tabellarische Auflistung aller detektierten Events eines Laufs. Jedes Event zeigt:
   * Typ, Dauer und betroffene Umweltfaktoren.
   * Das Highlight-Bild des Fehlers.
   * Einen grafischen Plot (Kennzahl über die Zeit / den Umweltfaktor), um Trends sichtbar zu machen (z. B. *„Anstieg des Flackerns ab Minute 45 bei steigendem Umweltfaktor“*).

---

## 6. Konfigurations-Schicht (Anti-Hardcoding-Leitlinie)

Sämtliche verhaltenssteuernden Parameter müssen strikt aus dem Quellcode herausgehalten und in einer zentralen Konfigurationsdatei (z. B. `config.yaml` oder `config.json`) deklariert werden.

**Geforderte Konfigurations-Parameter:**
* **Input-Konfiguration:** Pfad zur Videodatei / ID des Eingabegeräts.
* **Kalibrierungs-Parameter:** Dauer der Vorlaufzeit (Standard: 60s).
* **Toleranz-Schwellen:** Multiplikatoren für die dynamische Standardabweichung (z. B. `threshold_multiplier: 3.5`).
* **Debounce-Filter:** Mindestanzahl an Frames, die eine Anomalie aufweisen müssen, um ein Event auszulösen (Vermeidung von Singulär-Rauschen).
* **UI-Parameter:** Refresh-Intervall des Dashboards.

---

## 7. Anweisungen für den Entwicklungs-Agenten (Prompt-Leitplanken)

1. **Keep It Simple, Stupid (KISS):** Erstelle keine komplexen UI-Designs oder verschachtelten Klassen-Hierarchien über die geforderten Schnittstellen hinaus. Das Tool soll wie ein Werkzeug für Ingenieure wirken, nicht wie eine Consumer-App.
2. **Robustheit:** Der Ausfall einzelner Frames oder algorithmische Fehlklassifikationen (False Positives) dürfen niemals zum Absturz der gesamten Verarbeitungs-Pipeline führen. Implementiere ein striktes Exception-Handling innerhalb der Filter-Schleifen.
3. **Datenintegrität:** Stelle sicher, dass die SQLite-Datenbanktransaktionen threadsicher und blockierungsfrei geschrieben werden, um Datenverlust bei plötzlichem Abbruch des Programms zu verhindern.