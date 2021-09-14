<template>
  <div class="texts">
    <div class="texts__header">
      <div class="texts__header__out">
        <p>output 1</p>
        <t-checkbox :inline="true" label="loss" />
        <t-checkbox :inline="true" label="данные метрики" />
      </div>
      <div class="texts__header__out">
        <p>output 2</p>
        <t-checkbox :inline="true" label="loss" />
        <t-checkbox :inline="true" label="данные метрики" />
      </div>
      <button>Показать</button>
    </div>
    <div class="texts__content">
      <div class="inner">
        <div class="epochs">
          <Table :data="progressTable" />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Table from './Table.vue';
export default {
  name: 'Texts',
  components: {
    Table,
  },
  computed: {
    progressTable() {
      return this.$store.getters['trainings/getTrainData']('progress_table') || [];
    },
  },
};
</script>

<style lang="scss" scoped>
.texts {
  margin-bottom: 20px;
  &__header {
    display: flex;
    gap: 15px;
    align-items: center;
    &__out {
      flex-shrink: 0;
      > p {
        color: #a7bed3;
        font-weight: 600;
        font-size: 14px;
        line-height: 17px;
        margin-bottom: 15px;
      }
    }
    button {
      flex: 0 0 150px;
    }
  }
  &__content {
    margin-top: 10px;
    box-sizing: border-box;
  }
}
</style>

