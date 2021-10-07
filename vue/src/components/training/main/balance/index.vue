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
          <t-select-new small :list="sortOps" v-model="settings.sorted" @input="select" />
        </t-field>
      </div>
    </div>
    <div class="t-balance__graphs" v-if="(settings.show_train || settings.show_val) && Object.keys(dataDalance).length > 0">
      <template v-for="(layer, index) of dataDalance">
        <template v-for="(data, i) of filter(layer)">
          <component :is="type[data.type]" v-bind="data" :key="`sdsdsa_${i + index}`" />
        </template>
      </template>
    </div>
    <div class="t-balance__overlay">
      <LoadSpiner v-if="isLearning && Object.keys(dataDalance).length === 0" text="Загрузка данных..." />
    </div>
  </div>
</template>

<script>
import LoadSpiner from '@/components/forms/LoadSpiner';
import { mapGetters } from 'vuex';

export default {
  name: 't-balance',
  components: {
    Heatmap: () => import('../stats/Heatmap'),
    CorrelationHeatmap: () => import('../stats/CorrelationHeatmap'),
    Scatter: () => import('../stats/Scatter'),
    Histogram: () => import('../stats/Histogram'),
    Table: () => import('../stats/Table'),
    Graphic: () => import('../stats/Graphic'),
    LoadSpiner,
  },
  data: () => ({
    selected: [],
    sortOps: [
      { label: 'по имени', value: 'alphabetic' },
      { label: 'по увеличению', value: 'ascending' },
      { label: 'по убыванию', value: 'descending' },
    ],
    type: {
      heatmap: 'heatmap',
      'correlation heatmap': 'CorrelationHeatmap',
      scatter: 'scatter',
      'distribution histogram': 'histogram',
      histogram: 'histogram',
      table: 'table',
      graphic: 'graphic',
    },
  }),
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    dataDalance() {
      return this.$store.getters['trainings/getTrainData']('data_balance') || [];
    },
    settings: {
      set(value) {
        this.$store.dispatch('trainings/setObjectInteractive', { data_balance: value });
      },
      get() {
        return this.$store.getters['trainings/getObjectInteractive']('data_balance');
      },
    },
  },
  methods: {
    filter(layer) {
      const arr = [];
      if (this.settings.show_train) {
        arr.push('train');
      }
      if (this.settings.show_val) {
        arr.push('val');
      }
      return layer.filter(item => arr.includes(item.type_data));
    },
    change() {
      this.handleClick();
    },
    async handleClick() {
      // await this.$store.dispatch('trainings/interactive', {});
    },
    async select(sorted) {
      console.log(sorted);
      await this.$store.dispatch('trainings/interactive', {
        data_balance: {
          sorted,
          show_val: this.settings.show_val,
          show_train: this.settings.show_train,
        },
      });
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
  &__overlay {
    width: 100%;
    height: 100%;
    z-index: 5;
    display: flex;
    align-items: center;
    justify-content: center;
  }
}
</style>