from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "NYPD_Arrests_Data__Historic_.csv"
OUTPUT_DIR = BASE_DIR / "graficas"

# Cargar el dataset histórico de arrestos de NYPD.
df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

df["ARREST_DATE"] = pd.to_datetime(df["ARREST_DATE"], format="%m/%d/%Y")
mes_arresto = df["ARREST_DATE"].dt.to_period("M").dt.to_timestamp()
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Filas y columnas: {df.shape}")
print("Columnas:")
print(df.columns.tolist())
print("\nPrimeras filas:")
print(df.head())

print("\nValores faltantes por columna:")
print(df.isna().sum().sort_values(ascending=False).head(10))

arrestos_mes = mes_arresto.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 5))
arrestos_mes.plot(ax=ax, color="#176b87", linewidth=1.8)
ax.set_title("Cantidad de arrestos por mes")
ax.set_xlabel("Fecha")
ax.set_ylabel("Arrestos")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_arrestos_por_mes.png", dpi=150)
plt.close(fig)

arrestos_borough = df["ARREST_BORO"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
arrestos_borough.plot(kind="barh", ax=ax, color="#e07a5f")
ax.set_title("Arrestos por borough")
ax.set_xlabel("Arrestos")
ax.set_ylabel("Borough")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_arrestos_por_borough.png", dpi=150)
plt.close(fig)

top_delitos = df["OFNS_DESC"].value_counts().head(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
top_delitos.plot(kind="barh", ax=ax, color="#3a7d44")
ax.set_title("Diez descripciones de delito más frecuentes")
ax.set_xlabel("Arrestos")
ax.set_ylabel("Descripción")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_delitos_mas_frecuentes.png", dpi=150)
plt.close(fig)

perfil = pd.crosstab(df["AGE_GROUP"], df["PERP_SEX"])
orden_edades = ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]
perfil = perfil.reindex([edad for edad in orden_edades if edad in perfil.index])
fig, ax = plt.subplots(figsize=(10, 5))
perfil.plot(kind="bar", ax=ax, color={"F": "#d95f02", "M": "#1b9e77"})
ax.set_title("Arrestos por grupo de edad y sexo")
ax.set_xlabel("Grupo de edad")
ax.set_ylabel("Arrestos")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Sexo")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_perfil_edad_sexo.png", dpi=150)
plt.close(fig)

print(f"\nGráficas guardadas en: {OUTPUT_DIR}")
for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"- {path.name}")
