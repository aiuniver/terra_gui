<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-field inline :label="'Показать тренировочную выборку'">
            <t-checkbox-new name="show_train" v-model="settings.show_train" @change="change" />
          </t-field>
          <t-field inline :label="'Показать проверочную выборку'">
            <t-checkbox-new name="show_val" v-model="settings.show_val" @change="change" />
          </t-field>
        </div>
        <t-field inline :label="'Сортировать'">
          <t-select-new small :list="sortOps" v-model="settings.sorted" />
        </t-field>
        <t-button class="t-balance__btn" @click="handleClick">Показать</t-button>
      </div>
    </div>
    <div class="t-balance__graphs">
      <template v-for="(layer, index) of dataDalance">
        <template v-for="(item, i) of filter(layer)">
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
    // train: false,
    // test: false,
    selected: [],
    sortOps: [
      { label: 'по имени', value: 'alphabetic' },
      { label: 'по увеличению', value: 'ascending' },
      { label: 'по убыванию', value: 'descending' },
    ],
    // sortSelected: 'alphabetic',
  }),
  computed: {
    dataDalance() {
      return this.$store.getters['trainings/getTrainData']('data_balance') || [];
    },
    settings: {
      set(value) {
        this.$store.dispatch('trainings/setObjectInteractive', { 'data_balance': value })
      },
      get() {
        return this.$store.getters['trainings/getObjectInteractive']('data_balance')
      },
    },
  },
  methods: {
    filter(layer) {
      return layer.filter(item => this.selected.includes(item.type_data));
    },
    change({ name, value }) {
      const temp = name !== 'show_train' ? 'train' : 'val';
      if (!value) {
        this.selected = this.selected.filter(item => item !== temp);
      } else {
        this.selected.push(temp);
      }
      console.log(this.selected);
    },
    async handleClick() {
      // const data = {
      //   data_balance: {
      //     show_train: this.train,
      //     show_val: this.test,
      //     sorted: this.sortSelected,
      //   },
      // };
      await this.$store.dispatch('trainings/interactive', {});
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