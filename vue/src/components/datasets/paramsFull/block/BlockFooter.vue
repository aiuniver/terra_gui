<template>
  <div class="dataset-options">
    <div class="form">
      <div class="dataset-field">
        <label for="name">Название датасета</label>
        <input type="text" class="terra-input large-input" name="name" id="name">
      </div>
      <div class="dataset-field">
        <label for="tags">Теги</label>
        <div class="tags">
          <button type="button" class="add-btn"><i class="plus"></i><span>Добавить</span></button>
          <input type="text" class="terra-input" onkeydown="this.style.width = ((this.value.length + 1) * 8) + 'px';">
          <input type="text" class="terra-input" value="images" onkeydown="this.style.width = ((this.value.length + 1) * 8) + 'px';">
          <input type="text" class="terra-input" value="images" onkeydown="this.style.width = ((this.value.length + 1) * 8) + 'px';">
        </div>
      </div>
      <div class="dataset-field">
        <label for="slider">Train / Val / Test</label>
        <div class="slider">
          <DoubleSlider/>
        </div>
      </div>
      <div class="dataset-field">
        <label for="subsequence">Сохранить последовательность</label>
        <div class="checkout-switch">
          <input id="subsequence" type="checkbox" name="subsequence"/>
          <span class="switcher"></span>
        </div>
      </div>
      <div class="dataset-field">
        <label for="generator">Использовать генератор</label>
        <div class="checkout-switch">
          <input id="generator" type="checkbox" name="generator"/>
          <span class="switcher"></span>
        </div>
      </div>
    </div>
    <div class="action">
      <button>Сформировать</button>
    </div>
  </div>
</template>

<script>
import DoubleSlider from "@/components/forms/DoubleSlider";

export default {
  name: "BlockFooter",
  components: {
    DoubleSlider,
  },
};
</script>

<style lang="scss" scoped>
  .plus{
    width: 14px;
    height: 14px;
    margin-top: -2px;
    &:before{
      content: ' ';
      display: block;
      width: 2px;
      height: 14px;
      background: #A7BED3;
      margin-left: 6px;
    }

    &:after{
      content: ' ';
      display: block;
      width: 14px;
      height: 2px;
      background: #A7BED3;
      margin-top: -8px;
    }

  }
  .dataset-options{
    display: flex;
    justify-content: space-between;
  }
  .form{
    display: flex;
  }
  .dataset-field{
    padding: 16px 18px;
    font-size: 0.75rem;
    display: flex;
    flex-direction: column;
    label{
      color: #A7BED3;
    }
  }
  .large-input{
    max-width: 150px;
  }
  .terra-input{
    margin-top: 10px;
    height: 24px;
    color: #FFFFFF;
  }
  .tags{
    display: flex;
    font-size: 12px;
    .terra-input{
      margin-left: 10px;
      padding: 2px 4px;
      width: 60px;
      color: #A7BED3;
    }
  }
  .add-btn{
      background: #242F3D;
      height: 24px;
      width: 90px;
      padding: 2px 4px;
      margin-top: 10px;
      box-shadow: none;
      border-color: #6C7883;
      color: #A7BED3;
      display: flex;
      align-items: center;
      span{
        padding-left: 9px;
      }
    }
  .action{
    padding-top: 26px;
    padding-right: 20px;
    font-size: 0.8rem;
    float: right;
    button{
      padding: 8px 16px;
    }
  }
  .checkout-switch{
    margin-top: 10px;
  }
</style>