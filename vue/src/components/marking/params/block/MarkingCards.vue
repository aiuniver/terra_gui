<template>
<div class="t-marking">
    <div class="t-marking__header">
        <p>Список доступных задач</p>
        <MarkSelect
        :small="true"
        :inline="true"
        label="Сортировать по:"
        :lists="[]"
        select-pos="right"
        width="180px"
        />
    </div>
    <div class="t-marking__cards">
        <t-card-marking />
        <t-card-marking />
        <t-card-marking />
    </div>
</div>
</template>

<script>
import tCardMarking from '@/components/marking/params/components/card/CardMarking.vue';
import MarkSelect from '../components/Select.vue'

export default {
    name: 'marking-cards',
    components: {
        tCardMarking,
        MarkSelect
    }
}
</script>

<style lang="scss" scoped>
.t-marking {
    padding: 30px 20px;
    &__cards {
        margin-top: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    &__header {
      display: flex;
      justify-content: space-between;
      width: 100%;
      align-items: center;
      p {
        font-size: 14px;
        line-height: 24px;
      }
    }
}
</style>