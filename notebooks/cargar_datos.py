from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "NYPD_Arrests_Data__Historic_.csv"
OUTPUT_DIR = BASE_DIR / "graficas"
OUTPUT_DIR.mkdir(exist_ok=True)


def unique_lookup(frame, key_column, value_column):
    pairs = frame.loc[
        frame[key_column].notna() & frame[value_column].notna(),
        [key_column, value_column],
    ].drop_duplicates()
    unique_keys = pairs.groupby(key_column)[value_column].nunique()
    unique_keys = unique_keys[unique_keys.eq(1)].index
    return (
        pairs[pairs[key_column].isin(unique_keys)]
        .drop_duplicates(key_column)
        .set_index(key_column)[value_column]
    )


# Cargar el dataset histórico de arrestos de NYPD.
df = pd.read_csv(
    DATA_PATH,
    low_memory=False,
)

lookup_keys = {
    "PD_DESC": "PD_CD",
    "KY_CD": "PD_CD",
    "OFNS_DESC": "PD_CD",
    "LAW_CAT_CD": "PD_CD",
    "ARREST_BORO": "ARREST_PRECINCT",
}
lookups = {
    target: unique_lookup(df, key, target)
    for target, key in lookup_keys.items()
}
valores_completados = {}
for target, key in lookup_keys.items():
    inferido = df[key].map(lookups[target])
    faltante_recuperable = df[target].isna() & inferido.notna()
    valores_completados[target] = int(faltante_recuperable.sum())
    df.loc[faltante_recuperable, target] = inferido.loc[faltante_recuperable]

print("\nValores faltantes completados mediante correspondencias únicas:")
for columna, cantidad in valores_completados.items():
    if cantidad:
        print(f"{columna}: {cantidad:,}")

faltantes_restantes = df.isna().sum()
print("Faltantes restantes sin valor verificable:")
for columna, cantidad in faltantes_restantes[ faltantes_restantes.gt(0) ].items():
    print(f"{columna}: {cantidad:,}")

cleaned_path = DATA_PATH.with_name("NYPD_Arrests_Data__Historic__completado.csv")
df.to_csv(cleaned_path, index=False)
print(f"Copia con valores recuperables: {cleaned_path}")

grupos_edad = ["<18", "18-24", "25-44", "45-64", "65+", "UNKNOWN"]
edad_valida = df["AGE_GROUP"].isin(grupos_edad)
conteos_edad = (
    df.loc[edad_valida, "AGE_GROUP"]
    .value_counts()
    .reindex(grupos_edad, fill_value=0)
)
total_edad_valida = int(conteos_edad.sum())
valores_edad_invalidos = int((df["AGE_GROUP"].notna() & ~edad_valida).sum())
edades_sin_dato = int(df["AGE_GROUP"].isna().sum())

print("\nArrestos por grupo de edad:")
for grupo, cantidad in conteos_edad.items():
    porcentaje = cantidad / total_edad_valida * 100
    print(f"{grupo}: {cantidad:,} ({porcentaje:.2f}%)")
print(
    "Valores de edad excluidos del gráfico: "
    f"{valores_edad_invalidos:,} fuera de catálogo y {edades_sin_dato:,} sin dato."
)
arrestos_18_44 = int(conteos_edad.loc[["18-24", "25-44"]].sum())
print(
    "Los grupos de 18 a 44 años representan "
    f"{arrestos_18_44 / total_edad_valida * 100:.2f}% de los registros válidos."
)

fig, ax = plt.subplots(figsize=(9, 5))
colores_edad = ["#ff1493", "#f00078", "#e60073", "#d90068", "#cc005f", "#b80059"]
bars = ax.barh(grupos_edad, conteos_edad.values, color=colores_edad)
ax.invert_yaxis()
limite = int(conteos_edad.max())
ax.set_xlim(0, limite * 1.25)
for bar, cantidad in zip(bars, conteos_edad.values):
    ax.text(
        bar.get_width() + limite * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{cantidad:,} ({cantidad / total_edad_valida:.2%})",
        va="center",
    )
ax.set_title("Arrestos por grupo de edad")
ax.set_xlabel("Cantidad de arrestos")
ax.set_ylabel("Grupo de edad")
ax.grid(axis="x", alpha=0.25)
fig.text(
    0.1,
    0.01,
    f"Porcentajes sobre {total_edad_valida:,} registros con grupo válido; "
    f"se excluyen {valores_edad_invalidos + edades_sin_dato:,} filas.",
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUTPUT_DIR / "01_grupos_de_edad_rosa.png", dpi=150)
plt.close(fig)

edades_comparables = grupos_edad[:-1]
cruces_edad_delito = df.loc[
    df["AGE_GROUP"].isin(edades_comparables) & df["OFNS_DESC"].notna(),
    ["AGE_GROUP", "OFNS_DESC"],
]
top_10_delitos_edad = cruces_edad_delito["OFNS_DESC"].value_counts().head(10).index
conteos_edad_delito = pd.crosstab(
    cruces_edad_delito["OFNS_DESC"], cruces_edad_delito["AGE_GROUP"]
).reindex(index=top_10_delitos_edad, columns=edades_comparables, fill_value=0)
totales_delito_edad = cruces_edad_delito["AGE_GROUP"].value_counts().reindex(
    edades_comparables
)
porcentajes_edad_delito = conteos_edad_delito.div(totales_delito_edad, axis="columns") * 100

print("\nDelito más frecuente dentro de cada grupo de edad:")
for grupo in edades_comparables:
    delito = cruces_edad_delito.loc[
        cruces_edad_delito["AGE_GROUP"] == grupo, "OFNS_DESC"
    ].value_counts().idxmax()
    cantidad = int(conteos_edad_delito.loc[delito, grupo])
    porcentaje = cantidad / int(totales_delito_edad.loc[grupo]) * 100
    print(f"{grupo}: {delito} ({cantidad:,}; {porcentaje:.2f}% del grupo)")

fig, ax = plt.subplots(figsize=(12, 7))
cmap_rosa = LinearSegmentedColormap.from_list(
    "rosa_delitos", ["#fff1f6", "#ff8fc1", "#c90061"]
)
imagen = ax.imshow(porcentajes_edad_delito.values, cmap=cmap_rosa, aspect="auto")
ax.set_xticks(range(len(edades_comparables)), labels=edades_comparables)
ax.set_yticks(range(len(top_10_delitos_edad)), labels=top_10_delitos_edad)
ax.set_title("Diez delitos más frecuentes por grupo de edad")
ax.set_xlabel("Grupo de edad")
ax.set_ylabel("Descripción del delito")
for fila in range(len(top_10_delitos_edad)):
    for columna in range(len(edades_comparables)):
        porcentaje = porcentajes_edad_delito.iat[fila, columna]
        color_texto = "white" if porcentaje > porcentajes_edad_delito.values.max() * 0.55 else "#4a1830"
        ax.text(columna, fila, f"{porcentaje:.1f}%", ha="center", va="center", color=color_texto, fontsize=8)
fig.colorbar(imagen, ax=ax, label="Porcentaje de arrestos del grupo")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_edad_delito_rosa.png", dpi=150)
plt.close(fig)

sexos_comparables = ["F", "M"]
cruces_sexo_delito = df.loc[
    df["PERP_SEX"].isin(sexos_comparables) & df["OFNS_DESC"].notna(),
    ["PERP_SEX", "OFNS_DESC"],
]
top_10_delitos_sexo = cruces_sexo_delito["OFNS_DESC"].value_counts().head(10).index
conteos_sexo_delito = pd.crosstab(
    cruces_sexo_delito["OFNS_DESC"], cruces_sexo_delito["PERP_SEX"]
).reindex(index=top_10_delitos_sexo, columns=sexos_comparables, fill_value=0)
totales_delito_sexo = cruces_sexo_delito["PERP_SEX"].value_counts().reindex(
    sexos_comparables
)
porcentajes_sexo_delito = conteos_sexo_delito.div(totales_delito_sexo, axis="columns") * 100

print("\nDelito más frecuente dentro de cada sexo registrado:")
for sexo in sexos_comparables:
    delito = cruces_sexo_delito.loc[
        cruces_sexo_delito["PERP_SEX"] == sexo, "OFNS_DESC"
    ].value_counts().idxmax()
    cantidad = int(conteos_sexo_delito.loc[delito, sexo])
    porcentaje = cantidad / int(totales_delito_sexo.loc[sexo]) * 100
    print(f"{sexo}: {delito} ({cantidad:,}; {porcentaje:.2f}% del grupo)")

fig, ax = plt.subplots(figsize=(9, 7))
imagen = ax.imshow(porcentajes_sexo_delito.values, cmap=cmap_rosa, aspect="auto")
ax.set_xticks(range(len(sexos_comparables)), labels=["F (mujer)", "M (hombre)"])
ax.set_yticks(range(len(top_10_delitos_sexo)), labels=top_10_delitos_sexo)
ax.set_title("Diez delitos más frecuentes por sexo registrado")
ax.set_xlabel("Sexo registrado")
ax.set_ylabel("Descripción del delito")
for fila in range(len(top_10_delitos_sexo)):
    for columna in range(len(sexos_comparables)):
        porcentaje = porcentajes_sexo_delito.iat[fila, columna]
        color_texto = "white" if porcentaje > porcentajes_sexo_delito.values.max() * 0.55 else "#4a1830"
        ax.text(
            columna,
            fila,
            f"{porcentaje:.1f}%",
            ha="center",
            va="center",
            color=color_texto,
            fontsize=8,
        )
fig.colorbar(imagen, ax=ax, label="Porcentaje de arrestos del grupo")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_sexo_delito_rosa.png", dpi=150)
plt.close(fig)

df["ARREST_DATE"] = pd.to_datetime(df["ARREST_DATE"], format="%m/%d/%Y")
mes_arresto = df["ARREST_DATE"].dt.to_period("M").dt.to_timestamp()

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
fig.savefig(OUTPUT_DIR / "04_arrestos_por_mes.png", dpi=150)
plt.close(fig)

arrestos_borough = df["ARREST_BORO"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
arrestos_borough.plot(kind="barh", ax=ax, color="#e07a5f")
ax.set_title("Arrestos por borough")
ax.set_xlabel("Arrestos")
ax.set_ylabel("Borough")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_arrestos_por_borough.png", dpi=150)
plt.close(fig)

top_delitos = df["OFNS_DESC"].value_counts().head(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
top_delitos.plot(kind="barh", ax=ax, color="#3a7d44")
ax.set_title("Diez descripciones de delito más frecuentes")
ax.set_xlabel("Arrestos")
ax.set_ylabel("Descripción")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_delitos_mas_frecuentes.png", dpi=150)
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
fig.savefig(OUTPUT_DIR / "07_perfil_edad_sexo.png", dpi=150)
plt.close(fig)

print(f"\nGráficas guardadas en: {OUTPUT_DIR}")
for filename in [
    "01_grupos_de_edad_rosa.png",
    "02_edad_delito_rosa.png",
    "03_sexo_delito_rosa.png",
    "04_arrestos_por_mes.png",
    "05_arrestos_por_borough.png",
    "06_delitos_mas_frecuentes.png",
    "07_perfil_edad_sexo.png",
]:
    print(f"- {filename}")
