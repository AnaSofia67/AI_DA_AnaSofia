from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

top_5_delitos_edad = top_10_delitos_edad[:5]
porcentajes_top_5_edad = porcentajes_edad_delito.loc[top_5_delitos_edad]
colores_delitos = ["#ff1493", "#e60073", "#c90061", "#a90050", "#7f003f"]
ancho_barra = 0.15
fig, ax = plt.subplots(figsize=(14, 6))
for indice, delito in enumerate(top_5_delitos_edad):
    posiciones = [
        grupo + (indice - (len(top_5_delitos_edad) - 1) / 2) * ancho_barra
        for grupo in range(len(edades_comparables))
    ]
    ax.bar(
        posiciones,
        porcentajes_top_5_edad.loc[delito].values,
        width=ancho_barra,
        color=colores_delitos[indice],
        label=delito,
    )
ax.set_xticks(range(len(edades_comparables)), labels=edades_comparables)
ax.set_title("Histograma de frecuencias: edad y delito")
ax.set_xlabel("Grupo de edad")
ax.set_ylabel("Porcentaje de arrestos dentro del grupo")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.legend(title="Delito", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
fig.tight_layout(rect=(0, 0, 0.77, 1))
fig.savefig(OUTPUT_DIR / "02_histograma_edad_delito_rosa.png", dpi=150)
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
cmap_rosa = LinearSegmentedColormap.from_list(
    "rosa_delitos", ["#fff1f6", "#ff8fc1", "#c90061"]
)

print("\nDelito más frecuente dentro de cada sexo registrado:")
for sexo in sexos_comparables:
    delito = cruces_sexo_delito.loc[
        cruces_sexo_delito["PERP_SEX"] == sexo, "OFNS_DESC"
    ].value_counts().idxmax()
    cantidad = int(conteos_sexo_delito.loc[delito, sexo])
    porcentaje = cantidad / int(totales_delito_sexo.loc[sexo]) * 100
    print(f"{sexo}: {delito} ({cantidad:,}; {porcentaje:.2f}% del grupo)")

top_5_delitos_sexo = top_10_delitos_sexo[:5]
porcentajes_top_5_sexo = porcentajes_sexo_delito.loc[top_5_delitos_sexo]
fig, ax = plt.subplots(figsize=(11, 6))
posiciones = list(range(len(top_5_delitos_sexo)))
ancho_barra = 0.36
for sexo, desplazamiento, color in [
    ("F", -ancho_barra / 2, "#ff1493"),
    ("M", ancho_barra / 2, "#8f0056"),
]:
    porcentajes = porcentajes_top_5_sexo[sexo].values
    barras = ax.barh(
        [indice + desplazamiento for indice in posiciones],
        porcentajes,
        height=ancho_barra,
        color=color,
        label=sexo,
    )
    ax.bar_label(barras, labels=[f"{valor:.1f}%" for valor in porcentajes], padding=3, fontsize=8)
ax.set_yticks(posiciones, labels=top_5_delitos_sexo)
ax.invert_yaxis()
ax.set_title("Cinco delitos principales por sexo registrado")
ax.set_xlabel("Porcentaje dentro de cada grupo")
ax.set_ylabel("Descripción del delito")
ax.legend(title="Sexo registrado", labels=["F", "M"])
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_bar_chart_delito_sexo_rosa.png", dpi=150)
plt.close(fig)

filas_scatter = df.loc[
    df["AGE_GROUP"].isin(edades_comparables)
    & df["PERP_SEX"].isin(sexos_comparables),
    ["AGE_GROUP", "PERP_SEX", "OFNS_DESC"],
]
filas_scatter_validas = filas_scatter.dropna(subset=["OFNS_DESC"])
top_delitos_scatter = filas_scatter_validas["OFNS_DESC"].value_counts().head(5).index
pares_demograficos = pd.MultiIndex.from_product(
    [edades_comparables, sexos_comparables], names=["AGE_GROUP", "PERP_SEX"]
)
conteos_scatter = pd.crosstab(
    [filas_scatter_validas["AGE_GROUP"], filas_scatter_validas["PERP_SEX"]],
    filas_scatter_validas["OFNS_DESC"],
).reindex(index=pares_demograficos, columns=top_delitos_scatter, fill_value=0)
denominadores_scatter = filas_scatter_validas.groupby(
    ["AGE_GROUP", "PERP_SEX"]
).size().reindex(pares_demograficos)
porcentajes_scatter = conteos_scatter.div(denominadores_scatter, axis="index") * 100
totales_demograficos = filas_scatter.groupby(
    ["AGE_GROUP", "PERP_SEX"]
).size().reindex(pares_demograficos)
maximo_demografico = int(totales_demograficos.max())
tamanos_burbujas = 35 + 220 * totales_demograficos / maximo_demografico
colores_sexo = {"F": "#ff1493", "M": "#8f0056"}
marcadores_delito = ["o", "s", "^", "D", "P"]

fig, ax = plt.subplots(figsize=(14, 8))
for indice_delito, delito in enumerate(top_delitos_scatter):
    for sexo, desplazamiento in [("F", -0.12), ("M", 0.12)]:
        posiciones = [
            indice_edad + desplazamiento + (indice_delito - 2) * 0.025
            for indice_edad in range(len(edades_comparables))
        ]
        pares = [(edad, sexo) for edad in edades_comparables]
        ax.scatter(
            posiciones,
            [porcentajes_scatter.loc[par, delito] for par in pares],
            s=[tamanos_burbujas.loc[par] for par in pares],
            color=colores_sexo[sexo],
            marker=marcadores_delito[indice_delito],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

ax.set_xticks(
    range(len(edades_comparables)),
    labels=[f"{edad}\nn={int(conteos_edad.loc[edad]):,}" for edad in edades_comparables],
)
ax.set_xlim(-0.5, len(edades_comparables) - 0.5)
ax.set_ylim(0, float(porcentajes_scatter.values.max()) * 1.25)
ax.set_title("Scatterplot integrado: edad, delito y sexo")
ax.set_xlabel("Grupo de edad y total de arrestos")
ax.set_ylabel("Porcentaje dentro del grupo edad + sexo")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

leyenda_sexo = [
    Line2D(
        [0], [0], marker="o", linestyle="None", color=colores_sexo[sexo],
        label=sexo, markersize=8,
    )
    for sexo in sexos_comparables
]
leyenda_delitos = [
    Line2D(
        [0], [0], marker=marcadores_delito[indice], linestyle="None",
        color="#777777", markerfacecolor="#777777", label=delito, markersize=8,
    )
    for indice, delito in enumerate(top_delitos_scatter)
]
referencias_tamano = pd.Series(totales_demograficos).quantile([0.1, 0.5, 0.9])
leyenda_tamanos = [
    Line2D(
        [0], [0], marker="o", linestyle="None", color="#999999",
        markersize=(35 + 220 * cantidad / maximo_demografico) ** 0.5,
        label=f"{cantidad / 1_000:.0f}k",
    )
    for cantidad in referencias_tamano
]
leyenda = ax.legend(
    handles=leyenda_sexo,
    title="Color: sexo",
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
)
ax.add_artist(leyenda)
leyenda = ax.legend(
    handles=leyenda_delitos,
    title="Forma: delito",
    loc="upper left",
    bbox_to_anchor=(1.01, 0.78),
    fontsize=8,
)
ax.add_artist(leyenda)
ax.legend(
    handles=leyenda_tamanos,
    title="Tamaño: arrestos edad + sexo",
    loc="upper left",
    bbox_to_anchor=(1.01, 0.38),
    fontsize=8,
)
fig.text(
    0.1,
    0.015,
    "Solo se muestran los cinco delitos principales; el porcentaje usa registros con delito informado.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.05, 0.74, 1))
fig.savefig(OUTPUT_DIR / "08_scatterplot_tres_insights_rosa.png", dpi=150)
plt.close(fig)

valores_heatmap = porcentajes_scatter.T
etiquetas_heatmap = []
for edad in edades_comparables:
    for sexo in sexos_comparables:
        total_grupo = int(totales_demograficos.loc[(edad, sexo)])
        total_corto = (
            f"{total_grupo / 1_000_000:.1f}M"
            if total_grupo >= 1_000_000
            else f"{total_grupo / 1_000:.0f}k"
        )
        etiquetas_heatmap.append(f"{edad}\n{sexo}\nn={total_corto}")
fig, ax = plt.subplots(figsize=(16, 7))
imagen = ax.imshow(valores_heatmap.values, cmap=cmap_rosa, aspect="auto")
ax.set_xticks(range(len(etiquetas_heatmap)), labels=etiquetas_heatmap)
ax.tick_params(axis="x", labelsize=8)
ax.set_yticks(range(len(top_delitos_scatter)), labels=top_delitos_scatter)
ax.set_title("Heatmap integrado: edad, delito y sexo")
ax.set_xlabel("Edad y sexo registrados")
ax.set_ylabel("Cinco delitos más frecuentes")
for separador in range(1, len(edades_comparables)):
    ax.axvline(separador * len(sexos_comparables) - 0.5, color="white", linewidth=2)
for fila in range(len(top_delitos_scatter)):
    for columna in range(len(etiquetas_heatmap)):
        porcentaje = valores_heatmap.iat[fila, columna]
        color_texto = "white" if porcentaje > valores_heatmap.values.max() * 0.55 else "#4a1830"
        ax.text(
            columna,
            fila,
            f"{porcentaje:.1f}%",
            ha="center",
            va="center",
            color=color_texto,
            fontsize=8,
        )
fig.colorbar(imagen, ax=ax, label="Porcentaje dentro de edad + sexo")
fig.text(
    0.1,
    0.015,
    "n indica arrestos del grupo; los porcentajes excluyen descripciones de delito faltantes.",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(OUTPUT_DIR / "09_heatmap_tres_insights_rosa.png", dpi=150)
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

print(f"\nGráficas guardadas en: {OUTPUT_DIR}")
for filename in [
    "01_grupos_de_edad_rosa.png",
    "02_histograma_edad_delito_rosa.png",
    "03_bar_chart_delito_sexo_rosa.png",
    "04_arrestos_por_mes.png",
    "05_arrestos_por_borough.png",
    "06_delitos_mas_frecuentes.png",
    "08_scatterplot_tres_insights_rosa.png",
    "09_heatmap_tres_insights_rosa.png",
]:
    print(f"- {filename}")
