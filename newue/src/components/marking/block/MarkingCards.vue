<template>
  <div class="t-marking">
    <div class="t-marking__header">
      <p>Список доступных задач</p>
      <d-select :small="true" :inline="true" label="Сортировать по:" :lists="[]" select-pos="right" width="180px" />
    </div>
    <div class="t-marking__cards">
      <CardMarking />
      <CardMarking />
      <CardMarking />
    </div>
  </div>
</template>

<script>
import CardMarking from './CardMarking.vue';

export default {
  name: 'marking-cards',
  components: {
    CardMarking,
  },
};
</script>

<style lang="scss">
.t-marking {
  padding: 30px 20px;
  &__cards {
    margin-top: 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
  }
  &__header {
    display: flex;
    justify-content: space-between;
    width: 100%;
    align-items: center;
    p {
      font-size: 14px;
      line-height: 24px;
    }
  }
}
</style>