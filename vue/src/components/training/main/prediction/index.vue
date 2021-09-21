<template>
  <div class="predictions">
    <!-- <h3>Параметры</h3> -->
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Показать тренировочную выборку">
          <t-checkbox-new :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <t-select-new small />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <t-select-new small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Показать примеров">
          <t-input-new :value="10" type="number" small />
        </t-field>
        <t-field inline label="Показать статистику">
          <t-checkbox-new :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Автообновление">
          <t-checkbox-new small />
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
  computed: {
    predictData() {
      return this.$store.getters['trainings/getTrainData']('intermediate_result') || {};
    },
    train(){
      return this.$store.getters['trainings/getTrainDisplay'] || {}
    }
  },
  data: () => ({
    showTextTable: false,
  }),
  methods:{
    async show(){
      await this.$store.dispatch('trainings/interactive', {'intermediate_result':this.train['intermediate_result']})

    }
  }
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
