<template>
    <div class="t-balance">
        <div class="t-balance__header">
            <p>Параметры</p>
            <div class="t-balance__wrapper">
                <div class="t-balance__checks">
                    <t-checkbox :inline="true" label="Показать тренировочную выборку"/>
                    <t-checkbox :inline="true" label="Показать проверочную выборку"/>
                </div>
                <Select
                :small="true"
                :inline="true"
                label="Сортировать"
                :lists="[]"
                width="180px"
                ></Select>
                <button>Показать</button>
            </div>
        </div>
        <div class="t-balance__graphs">
            <Graph label="Тренировочная выборка" :values="[5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]" :step="{ num: 5, value: 1000 }"/>
            <Graph label="Проверочная выборка" :values="[194, 203, 187, 199, 184, 189, 211, 200, 224, 209]" :step="{ num: 4, value: 50 }"/>
        </div>
    </div>
</template>

<script>
import Select from './Select.vue'
import Graph from './Graph.vue'

export default {
    name: 't-balance',
    components: {
        Select,
        Graph
    }
}
</script>

<style lang="scss" scoped>
.t-balance {
    &__graphs {
        display: flex;
        flex-wrap: wrap;
        gap: 30px;
        margin-top: 20px;
    }
    &__wrapper {
        display: flex;
        gap: 20px;
    }
    p {
        color: #A7BED3;
        font-size: 14px;
        line-height: 17px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    button {
        width: 150px;
    }
}
</style>