<template>
  <div class="block-footer">
    <div class="block-footer__item">
      <t-input name="name" small>
        Название датасета
      </t-input>
    </div>
    <div class="block-footer__item block-tags">
      <TTags />
    </div>
    <div class="block-footer__item">
      <DoubleSlider />
    </div>
    <div class="block-footer__item">
      <t-checkbox name="subsequence">Сохранить последовательность</t-checkbox>
    </div>
    <div class="block-footer__item">
      <t-checkbox name="use_generator">Использовать генератор</t-checkbox>
    </div>
    <div class="action">
      <t-button>Сформировать</t-button>
    </div>
  </div>
</template>

<script>
import DoubleSlider from '@/components/forms/DoubleSlider';
import TTags from '@/components/forms/TTags';

export default {
  name: 'BlockFooter',
  components: {
    DoubleSlider,
    TTags,
  },
};
</script>

<style lang="scss" scoped>
.block-footer {
  display: flex;
  justify-content: space-between;
  padding: 22px 24px;
  &__item {
    flex: 0 0 150px;
    margin-right: 36px;
  }
  &__item.block-tags{
    flex: 0 0 250px;
  }
}


.action {
  font-size: 0.8rem;
  button {
    padding: 8px 16px;
  }
}

</style>