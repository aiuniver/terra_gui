<template>
  <div class="predictions">
    <!-- <h3>Параметры</h3> -->
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <t-select-new :list="sortData" v-model="checks.example_choice_type" small />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <t-select-new :list="sortOutput" v-model="checks.main_output" small />
        </t-field>
        <t-field inline label="Показать примеров">
          <t-input-new v-model.number="checks.num_examples" type="number" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Выводить промежуточные результаты">
          <t-checkbox-new v-model="checks.show_results" small />
        </t-field>
        <t-field inline label="Показать статистику">
          <t-checkbox-new v-model="checks.show_statistic" :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Автообновление">
          <t-checkbox-new v-model="checks.autoupdate" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-button style="width: 150px" @click.native="show">Показать</t-button>
      </div>
    </div>
    <div class="predictions__body" >
      <!-- <div class="predictions__overlay" v-if="loading || Object.keys(predictData).length === 0">
        <LoadSpiner :text="'Получение данных...'" />
      </div> -->
      <PredictTable :predict="predictData" />
    </div>
  </div>
</template>

<script>
import PredictTable from './PredictTable';
// import LoadSpiner from '@/components/forms/LoadSpiner';

export default {
  name: 'Predictions',
  components: {
    PredictTable,
    // LoadSpiner,
  },
  props: {
    outputs: Array,
    interactive: Object,
  },
  data: () => ({
    checks: {
      autoupdate: false,
      show_statistic: false,
      num_examples: 10,
      show_results: false,
      example_choice_type: 'seed',
      main_output: 2,
    },
    loading: true,
    showPredict: false,
    sortOutput: [],
    sortData: [
      { label: 'Best', value: 'best' },
      { label: 'Worst', value: 'worst' },
      { label: 'Seed', value: 'seed' },
      { label: 'Random', value: 'random' },
    ],
    showTextTable: false,
  }),
  created() {
    this.sortOutput = this.outputs.map(el => {
      return {
        label: `Выходной слой ${el.id}`,
        value: el.id,
      };
    });
  },
  computed: {
    predictData() {
      return this.$store.getters['trainings/getTrainData']('intermediate_result') || {};
    },
    statusTrain() {
      return this.$store.getters['trainings/getStatusTrain'];
    },
  },
  methods: {
    async show() {
      await this.$store.dispatch('trainings/interactive', {
        intermediate_result: { ...this.checks },
      });
      if (this.checks.show_results) {
        this.showPredict = true;
      }
      this.loading = false;
    },
  },
};
</script>

<style lang="scss" scoped>
.predictions {
  position: relative;
  &__overlay {
    position: absolute;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    background-color: rgb(14 22 33 / 30%);
    z-index: 5;
    top: 0;
    left: 0;
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
