<template>
    <div class="t-table">
        <p class="t-table__title">Выходной слой «3» – Classification report</p>
        <table class="t-table__table">
            <thead>
                <tr class="bold">
                    <th rowspan="2">Класс</th>
                    <th colspan="4">Статистические данные</th>
                </tr>
                <tr>
                    <th>Точность</th>
                    <th>Чувствительность</th>
                    <th>F1-мера</th>
                    <th>Количество</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(value, key, idx) in data_table" :key="idx">
                    <td>{{ key }}</td>
                    <td>{{ value['Точность'] }}</td>
                    <td>{{ value['Чувствительность'] }}</td>
                    <td>{{ value['F1-мера'] }}</td>
                    <td>{{ value['Количество'] }}</td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
export default {
    name: 't-table',
    data: () => ({
        data_table: {
            '<s1>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            '<s2>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            '<s3>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            '<s4>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            '<s5>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            '<s6>': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            'micro avg': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            'macro avg': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            'weighted avg': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
            'samples avg': {
                'Точность': 48.7,
                'Чувствительность': 99.2,
                'F1-мера': 64.1,
                'Количество': 90493,
            },
        }
    })
}
</script>

<style lang="scss" scoped>
.t-table {
    &__title {
        font-size: 14px;
        line-height: 17px;
        font-weight: 600;
        text-align: center;
        color: #A7BED3;
        margin: 10px auto;
    }
    .bold th {
        font-weight: 600;
        line-height: 16px;
    }
    &__table {
        font-size: 12px;
        line-height: 14px;
        border: 2px solid #242F3D;
        border-radius: 4px;
        overflow: hidden;
        thead {
            color: #A7BED3;
            background: #242F3D;
            height: 45px;
            th {
                text-align: center;
                font-weight: 400;
                min-width: 88px;
                padding: 2px 3px;
                border: 1px solid black;
            }
        }
        tbody {
            line-height: 16px;
            background: #17212B;
            td {
                text-align: center;
                padding: 4px 0;
                border: 1px solid black;
            }
            tr:nth-last-child(-n+4) {
                background: #242F3D;
            }
        }
    }
}
</style>