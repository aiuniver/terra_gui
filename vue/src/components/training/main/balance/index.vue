<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-field inline :label="'Показать тренировочную выборку'">
            <t-checkbox-new />
          </t-field>
          <t-field inline :label="'Показать проверочную выборку'">
            <t-checkbox-new small />
          </t-field>
        </div>
        <t-field inline :label="'Сортировать'">
          <t-select-new small :list="[]" />
        </t-field>
        <t-button class="t-balance__btn">Показать</t-button>
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
import Graph from './Graph';

export default {
  name: 't-balance',
  components: {
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
    height: 300px;
  }
  &__btn {
      margin-left: auto;
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