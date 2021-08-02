<template>
  <div class="dataset-options">
    <div class="dataset-field">
      <label for="name">Название датасета</label>
      <input type="text" class="terra-input large-input" name="name" id="name">
    </div>
    <div class="dataset-field">
      <label for="tags">Теги</label>
      <div class="tags">
        <button type="button" class="add-btn"><span>Добавить</span></button>
        <input type="text" class="terra-input">
        <input type="text" class="terra-input" value="images">
        <input type="text" class="terra-input" value="images">
      </div>
    </div>
    <div class="dataset-field">
      <label for="slider">Train / Val / Test</label>
      <div class="slider"></div>
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
    <div class="action">
      <button>Сформировать</button>
    </div>
  </div>
</template>
<style lang="scss" scoped>
  .dataset-options{
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
    }
    .add-btn{
      background: #242F3D;
      height: 24px;
      padding: 2px 4px;
      margin-top: 10px;
      box-shadow: none;
      border-color: #6C7883;
      color: #A7BED3;
    }
  }
  .action{
    padding-top: 26px;
    font-size: 0.8rem;
    button{
      padding: 8px 16px;
    }
  }
  .checkout-switch{
    margin-top: 10px;
  }
</style>