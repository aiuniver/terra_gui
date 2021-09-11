<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <p>Параметры</p>
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-checkbox :inline="true" label="Показать тренировочную выборку" />
          <t-checkbox :inline="true" label="Показать проверочную выборку" />
        </div>
        <Select :small="true" :inline="true" label="Сортировать" :lists="[]" width="180px"></Select>
        <button>Показать</button>
      </div>
    </div>
    <div class="t-balance__graphs">
      <template v-for="(id, index) of dataDalance">
        <template v-for="(item, i) of id">
          <Graph :key="'graph_' + index + '/' + i" v-bind="item" />
        </template>
      </template>
    </div>
  </div>
</template>

<script>
import Select from './Select.vue';
import Graph from './Graph.vue';

export default {
  name: 't-balance',
  components: {
    Select,
    Graph,
  },
  computed: {
    dataDalance() {
      return this.$store.getters['trainings/getTrainData']('data_balance') || [];
    },
  },
};
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
    color: #a7bed3;
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