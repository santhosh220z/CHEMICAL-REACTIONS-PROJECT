# Graph Report - .  (2026-08-18)

## Corpus Check
- Corpus is ~5,476 words - fits in a single context window. You may not need a graph.

## Summary
- 47 nodes · 63 edges · 12 communities (8 shown, 4 thin omitted)
- Extraction: 87% EXTRACTED · 11% INFERRED · 2% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Flask Route Handlers
- Local Storage Auth
- Model & Dataset
- Frontend Pages & Forms
- Background Image
- Prediction API Routes
- Web App & Data Prep
- Prediction Form Inputs
- Matplotlib
- NumPy
- Seaborn

## God Nodes (most connected - your core abstractions)
1. `Chemical Reaction Success Prediction Web App` - 10 edges
2. `Index (Prediction) Page` - 7 edges
3. `Artificial Neural Network (ANN)` - 6 edges
4. `About Page` - 6 edges
5. `Flask Backend` - 4 edges
6. `Local Storage Login & Registration` - 4 edges
7. `Login Page` - 4 edges
8. `getUsers() localStorage Reader` - 4 edges
9. `saveUsers() localStorage Writer` - 4 edges
10. `reactions.csv Dataset` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Chemical Reaction Success Prediction Web App` --references--> `TensorFlow`  [EXTRACTED]
  README.md → requirements.txt
- `fetch('/predict') Request` --references--> `Flask Backend`  [INFERRED]
  templates/index.html → README.md
- `Chemical Reaction Success Prediction Web App` --references--> `Pandas`  [EXTRACTED]
  README.md → requirements.txt
- `Chemical Reaction Success Prediction Web App` --references--> `scikit-learn`  [EXTRACTED]
  README.md → requirements.txt
- `Artificial Neural Network (ANN)` --references--> `TensorFlow`  [EXTRACTED]
  README.md → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Reaction Success Prediction Pipeline** — templates_index_prediction_form, templates_index_fetch_predict, readme_flask_backend, readme_ann_model, templates_index_prediction_result [INFERRED 0.85]
- **LocalStorage Demo Auth System** — templates_login_getusers, templates_registrion_saveusers, templates_registrion_users_storage, readme_localstorage_auth [INFERRED 0.85]
- **ML Technology Stack** — requirements_tensorflow, requirements_scikit_learn, requirements_pandas, readme_ann_model, readme_reactions_dataset [INFERRED 0.85]
- **Web App Background Styling Asset** — bg_background_image, bg_background_role_webapp [INFERRED 0.85]

## Communities (12 total, 4 thin omitted)

### Community 0 - "Flask Route Handlers"
Cohesion: 0.43
Nodes (7): about(), contact(), index(), login(), predict(), route, register()

### Community 1 - "Local Storage Auth"
Cohesion: 0.39
Nodes (8): Local Storage Login & Registration, getUsers() localStorage Reader, Login Form (loginForm), Login Page, Registration Form (regForm), Registration Page, saveUsers() localStorage Writer, users localStorage Key

### Community 2 - "Model & Dataset"
Cohesion: 0.47
Nodes (6): Artificial Neural Network (ANN), Flask Backend, reactions.csv Dataset, Team 2, TensorFlow, About Page

### Community 3 - "Frontend Pages & Forms"
Cohesion: 0.47
Nodes (6): Contact Form (name/email/message), Contact Page, fetch('/predict') Request, Index (Prediction) Page, Prediction Result Div, Thank You Page

### Community 4 - "Background Image"
Cohesion: 0.50
Nodes (4): Background Image Asset bg.jpg, Web App Background Image, Chemistry / Lab Themed Visual Content (unverified), Visual Content Could Not Be Verified (no vision support)

### Community 5 - "Prediction API Routes"
Cohesion: 0.67
Nodes (3): home(), predict(), route

### Community 6 - "Web App & Data Prep"
Cohesion: 0.67
Nodes (4): Chemical Reaction Success Prediction Web App, StandardScaler Input Scaling, Pandas, scikit-learn

## Ambiguous Edges - Review These
- `Background Image Asset bg.jpg` → `Chemistry / Lab Themed Visual Content (unverified)`  [AMBIGUOUS]
  bg.jpg · relation: conceptually_related_to

## Knowledge Gaps
- **6 isolated node(s):** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Web App Background Image` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Background Image Asset bg.jpg` and `Chemistry / Lab Themed Visual Content (unverified)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Index (Prediction) Page` connect `Frontend Pages & Forms` to `Prediction Form Inputs`, `Local Storage Auth`, `Model & Dataset`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `Chemical Reaction Success Prediction Web App` connect `Web App & Data Prep` to `Prediction Form Inputs`, `Local Storage Auth`, `Model & Dataset`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `Local Storage Login & Registration` connect `Local Storage Auth` to `Model & Dataset`, `Web App & Data Prep`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **What connects `Pandas`, `NumPy`, `Matplotlib` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._