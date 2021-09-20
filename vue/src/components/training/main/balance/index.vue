<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-field inline :label="'Показать тренировочную выборку'">
            <t-checkbox-new v-model="train" />
          </t-field>
          <t-field inline :label="'Показать проверочную выборку'">
            <t-checkbox-new v-model="test" small />
          </t-field>
        </div>
        <t-field inline :label="'Сортировать'">
          <t-select-new small :list="sortOps" v-model="sortSelected"/>
        </t-field>
        <t-button class="t-balance__btn" @click="handleClick">Показать</t-button>
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
  data: () => ({
    train: false,
    test: false,
    sortOps: [
      { label: 'по имени', value: 'alphabetic' },
      { label: 'по увеличению', value: 'ascending' },
      { label: 'по убыванию', value: 'descending' }
    ],
    sortSelected: 'alphabetic'
  }),
  computed: {
    dataDalance() {
      return this.$store.getters['trainings/getTrainData']('data_balance') || [];
    }
  },
  methods: {
    async handleClick() {
      const data = {
        "data_balance": {
          "show_train": this.train,
          "show_val": this.test,
          "sorted": this.sortSelected
        }
      }
      this.$store.dispatch('trainings/setTrainDisplay', data)

      await this.$store.dispatch('trainings/interactive', this.$store.getters['trainings/getTrainDisplay'])
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