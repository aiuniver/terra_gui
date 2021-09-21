<template>
  <div class="predictions">
    <!-- <h3>Параметры</h3> -->
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Показать тренировочную выборку">
          <t-checkbox-new v-model="checks.show_results" :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <t-select-new :list="sortData" v-model="checks.example_choice_type" small />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <t-select-new :list="sortOutput" v-model="checks.main_output" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Показать примеров">
          <t-input-new v-model.number="checks.num_examples" type="number" small />
        </t-field>
        <t-field inline label="Показать статистику">
          <t-checkbox-new v-model="checks.show_statistic" small />
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
    <TextTable :show="showTextTable" :predict="predictData" />
  </div>
</template>

<script>
import TextTable from './TextTableTest';
export default {
  name: 'Predictions',
  components: {
    TextTable,
  },
  props: {
    outputs: Array,
    interactive: Object,
  },
  data: () => ({
    predictData: {},
    checks: {
      autoupdate: false,
      show_statistic: false,
      num_examples: 10,
      show_results: false,
      example_choice_type: 'seed',
      main_output: 2,
    },
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
  methods: {
    async show() {
      const { data, status } = await this.$store.dispatch('trainings/interactive', {
        intermediate_result: { ...this.checks },
      });
      if (this.checks.show_results) {
        if (status && data) {
          this.predictData = data;
          this.showTextTable = true;
        }
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.predictions {
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
