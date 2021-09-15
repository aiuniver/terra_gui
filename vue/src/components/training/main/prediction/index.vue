<template>
  <div class="predictions">
    <h3>Параметры</h3>
    <div class="predictions__params">
      <div class="predictions__param">
        <t-field inline label="Показать тренировочную выборку">
          <t-checkbox-new :value="true" small />
        </t-field>
      </div>
      <div class="predictions__param">
        <t-field inline label="Данные для расчета">
          <t-select-new small style="width: 180px" />
        </t-field>
        <t-field inline label="Тип выбора данных">
          <t-select-new small style="width: 180px" />
        </t-field>
      </div>
      <div class="predictions__param">
        <!-- <Select :small="true" :inline="true" label="Сортировать" :lists="[]" width="180px"></Select> -->
        <t-field inline label="Показать примеров">
          <t-input-new style="width: 32px; text-align: center" :value="10" type="number" small />
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
        <t-button style="width: 150px; height: 40px" @click.native="showTextTable = !showTextTable">Показать</t-button>
      </div>
    </div>
    <TextTable :show="showTextTable" :predict="predictData"/>
  </div>
</template>

<script>
import TextTable from './TextTableTest';
import {mapGetters} from "vuex";
export default {
  name: 'Predictions',
  components: {
    TextTable,
  },
  data: () => ({
    showTextTable: false,
  }),
  computed: {
    ...mapGetters({
      predictData: 'trainings/getPredict',
    }),
  },
};
</script>

<style lang="scss" scoped>
.predictions__params {
  display: flex;
  margin-top: 10px;
  margin-bottom: 10px;
}
.predictions__param {
  margin-right: 30px;
  height: 100%;
}
</style>
