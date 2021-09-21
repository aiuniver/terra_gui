<template>
    <div class="t-graphic">
        <div class="t-graphic__title">Выходной слой «3»</div>
        <Plotly class="t-graphic__plotly" :data="data" :layout="layout" :display-mode-bar="false" />
    </div>
</template>

<script>
import { Plotly } from 'vue-plotly';

export default {
    name: 't-graphic',
    components: {
        Plotly
    },
    data: () => ({
        data: [{
            type: 'scatter',
            x: [1,2,3,4,5,6,7],
            y: [10,15,13,17,20,15,23],
            mode: 'lines',
            name: 'Правильные данные',
            line: {
                width: 3,
                color: '#89D764'
            }
        },
        {
            type: 'scatter',
            x: [2,3,4,5,6,7,8],
            y: [10,15,13,17,20,15,23],
            mode: 'lines',
            name: 'Предсказанные данные',
            line: {
                width: 3,
                color: '#FFB054'
            }
        }],
        layout: {
            width: 636,
            height: 352,
            plot_bgcolor: '#fff0',
            paper_bgcolor: '#242F3D',
            showlegend: true,
            legend: {
                orientation: 'h',
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
                    text: 'Время',
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
                    text: 'Значение',
                    font: { size: 12 }
                }
            }
        }
    })
}
</script>

<style lang="scss" scoped>
.t-graphic {
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