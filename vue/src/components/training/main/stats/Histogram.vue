<template>
    <div class="t-histogram">
        <p class="t-histogram__title">Выходной слой «3» – Распределение абсолютной ошибки </p>
        <Plotly class="t-histogram__plotly" :data="data" :layout="layout" :display-mode-bar="false" />
    </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
    name: 't-histogram',
    components: {
        Plotly
    },
    data: () => ({
        data: [{
            type: 'histogram',
            x: [1,2,3,4,5,6,7,2,3,4,5,6,7,1,2,3,4,5,6,7],
            y: [10,15,13,17,20,15,23,10,15,13,17,20,15,23,15,13,17,20,15,23],
            name: 'Регрессия',
        }],
        layout: {
            width: 636,
            height: 352,
            plot_bgcolor: '#fff0',
            paper_bgcolor: '#242F3D',
            showlegend: true,
            bargap: 0.1,
            legend: {
                orientation: 'h',
                yanchor: 'top'
            },
            font: {
                color: '#A7BED3',
                size: 12
            },
            margin: {
                pad: 1,
                t: 10,
                r: 20,
                b: 70,
                l: 60
            },
            xaxis: {
                gridcolor: '#17212B',
                showline: true,
                linecolor: '#A7BED3',
                linewidth: 1,
                title: {
                    text: 'Истинные значения',
                    standoff: 0,
                    font: { size: 12 }
                }
            },
            yaxis: {
                gridcolor: '#17212B',
                showline: true,
                linecolor: '#A7BED3',
                linewidth: 1,
                title: {
                    text: 'Предсказанные значения',
                    font: { size: 12 }
                }
            },
        }
    })
}
</script>

<style lang="scss" scoped>
.t-histogram {
    &__title {
        font-size: 14px;
        line-height: 17px;
        font-weight: 600;
        text-align: center;
        color: #A7BED3;
        margin: 10px auto;
    }
    &__plotly {
        border: 1px solid #6C7883;
        border-radius: 4px;
    }
}
</style>