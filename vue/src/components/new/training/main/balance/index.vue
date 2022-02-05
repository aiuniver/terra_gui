<template>
  <div class="t-balance">
    <div class="t-balance__header">
      <div class="t-balance__wrapper">
        <div class="t-balance__checks">
          <t-field inline :label="'Показать тренировочную выборку'">
            <TCheckbox name="show_train" v-model="settings.show_train" @change="change" />
          </t-field>
          <t-field inline :label="'Показать проверочную выборку'">
            <TCheckbox name="show_val" v-model="settings.show_val" @change="change" />
          </t-field>
        </div>
        <t-field inline :label="'Сортировать'">
          <TSelect small :list="sortOps" v-model="settings.sorted" @input="select" />
        </t-field>
      </div>
    </div>
    <div
      class="t-balance__graphs"
      v-if="(settings.show_train || settings.show_val) && Object.keys(dataDalance).length > 0"
    >
      <template v-for="({ train, val }, index) of filter(dataDalance)">
        <component v-if="train" :is="train.type" v-bind="train" :key="`train_${index}`" />
        <component v-if="val" :is="val.type" v-bind="val" :key="`val_${index}`" />
      </template>
    </div>
    <div class="t-balance__overlay">
      <LoadSpiner v-if="isLearning && Object.keys(dataDalance).length === 0" text="Загрузка данных..." />
    </div>
  </div>
</template>

<script>
import LoadSpiner from '@/components/forms/LoadSpiner';
import TCheckbox from '@/components/new/forms/TCheckbox';
import TSelect from '@/components/new/forms/TSelect';
import { mapGetters } from 'vuex';

export default {
  name: 't-balance',
  components: {
    Colormap: () => import('../stats/Colormap'),
    Heatmap: () => import('../stats/Heatmap'),
    Corheatmap: () => import('../stats/Corheatmap'),
    Scatter: () => import('../stats/Scatter'),
    Histogram: () => import('../stats/Histogram'),
    Bar: () => import('../stats/Histogram'),
    Table: () => import('../stats/STable'),
    Graphic: () => import('../stats/Graphic'),
    LoadSpiner,
    TCheckbox,
    TSelect
  },
  data: () => ({
    selected: [],
    sortOps: [
      { label: 'по имени', value: 'alphabetic' },
      { label: 'по увеличению', value: 'ascending' },
      { label: 'по убыванию', value: 'descending' },
    ],
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
      return layer.map(({ val, train }) => {
        const obj = {};
        if (this.settings.show_train) {
          obj.train = train;
        }
        if (this.settings.show_val) {
          obj.val = val;
        }
        return obj;
      });
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