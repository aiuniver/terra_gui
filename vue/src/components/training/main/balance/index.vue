<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-field inline :label="'Показать тренировочную выборку'">
            <t-checkbox-new v-model="graph1" />
          </t-field>
          <t-field inline :label="'Показать проверочную выборку'">
            <t-checkbox-new v-model="graph2" small />
          </t-field>
        </div>
        <t-field inline :label="'Сортировать'">
          <t-select-new small :list="[]" />
        </t-field>
        <t-button class="t-balance__btn" @click="handleClick">Показать</t-button>
      </div>
    </div>
    <div class="t-balance__graphs">
      <template v-for="(item, i) of showGraphs">
        <Graph :key="'graph_' + i + '/' + i" v-bind="item" />
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
  data: () => ({
    graph1: false,
    graph2: false
  }),
  computed: {
    dataDalance() {
      return this.$store.getters['trainings/getTrainData']('data_balance') || [];
    },
    showGraphs() {
      return this.dataDalance[2].filter(item => this[`graph${item.id}`])
    }
  },
  methods: {
    async handleClick() {
      const res = await this.$store.dispatch('trainings/interactive', {})
      console.log(`response`, res);
    }
  }
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