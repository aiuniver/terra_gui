<template>
  <div class="t-texts">
    <div class="t-texts__header">
      <div class="t-texts__block">
        <p>output 1</p>
        <t-field inline label="loss">
          <t-checkbox-new />
        </t-field>
        <t-field inline label="metrics">
          <t-checkbox-new />
        </t-field>
      </div>
      <div class="t-texts__block">
        <t-button>Показать</t-button>
      </div>
    </div>
    <div class="t-texts__content">
      <Table :data="data" />
    </div>
  </div>
</template>

<script>
import Table from './Table';
export default {
  name: 'Texts',
  components: {
    Table,
  },
  data: () => ({
    ops: {
      scrollPanel: {
        scrollingX: false,
        scrollingY: false,
      },
    },
  }),
  computed: {
    data() {
      return this.$store.getters['trainings/getTrainData']('progress_table') || [];
    },
  },
};
</script>

<style lang="scss" scoped>
.t-texts {
  margin-bottom: 20px;
  position: relative;
  &__header {
    display: flex;
    gap: 15px;
    align-items: center;
  }
  &__block {
    p {
      margin: 0 0 5px 0;
    }
    &:last-child {
      width: 150px;
      margin-left: auto;
    }
  }
  &__content {
    display: flex;
    height: 100%;
  }
}
</style>

