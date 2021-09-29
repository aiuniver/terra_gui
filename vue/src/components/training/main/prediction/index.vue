<template>
  <div class="predictions">
    <!-- <h3>Параметры</h3> -->
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <t-select-new :list="sortData" v-model="settings.example_choice_type" small />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <t-select-new :list="sortOutput" v-model="settings.main_output" small />
        </t-field>
        <t-field inline label="Показать примеров">
          <t-input-new v-model.number="settings.num_examples" type="number" small style="width: 109px;" />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Выводить промежуточные результаты">
          <t-checkbox-new v-model="settings.show_results" small />
        </t-field>
        <t-field inline label="Показать статистику">
          <t-checkbox-new v-model="settings.show_statistic" :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Автообновление">
          <t-checkbox-new v-model="settings.autoupdate" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-button style="width: 150px" @click.native="show">Показать</t-button>
      </div>
    </div>
    <div class="predictions__body">
      <PredictTable v-if="isEmpty" :predict="predictData" />
      <div v-else class="predictions__overlay">
        <LoadSpiner v-if="start && isLearning" text="Загрузка данных..." />
      </div>
    </div>
  </div>
</template>

<script>
import PredictTable from './PredictTable';
import LoadSpiner from '@/components/forms/LoadSpiner';
import { mapGetters } from 'vuex';

export default {
  name: 'Predictions',
  components: {
    PredictTable,
    LoadSpiner,
  },
  props: {
    outputs: Array,
    interactive: Object,
  },
  data: () => ({
    start: false,
    sortData: [
      { label: 'Best', value: 'best' },
      { label: 'Worst', value: 'worst' },
      { label: 'Seed', value: 'seed' },
      { label: 'Random', value: 'random' },
    ],
  }),
  computed: {
    ...mapGetters({
      status: 'trainings/getStatus',
    }),
    isLearning() {
      return ['addtrain', 'training'].includes(this.status);
    },
    isEmpty() {
      return Object.keys(this.predictData).length;
    },
    sortOutput() {
      return this.outputs.map(item => {
        return {
          label: `Выходной слой ${item.id}`,
          value: item.id,
        };
      });
    },
    settings: {
      set(value) {
        this.$store.dispatch('trainings/setObjectInteractive', { intermediate_result: value });
      },
      get() {
        return this.$store.getters['trainings/getObjectInteractive']('intermediate_result');
      },
    },
    predictData() {
      return this.$store.getters['trainings/getTrainData']('intermediate_result') || {};
    },
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
  },
  methods: {
    async show() {
      this.start = this.settings.show_results
      await this.$store.dispatch('trainings/interactive', {});
    },
  },
  created() {
    this.start = this.settings.show_results
  }
};
</script>

<style lang="scss" scoped>
.predictions {
  position: relative;
  &__body {
    position: relative;
    width: 100%;
  }
  &__overlay {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    z-index: 5;
  }
  &__params {
    display: flex;
    margin-top: 10px;
    margin-bottom: 10px;
  }
  &__param {
    padding: 0 10px 0 0;
    height: 100%;
    &:last-child {
      margin-left: auto;
      padding: 0;
    }
  }
}
</style>
