<template>
  <div class="project project-create" @click="$emit('addCard')">
    <div class="project-create-plus">
      <svg width="82" height="83" viewBox="0 0 82 83" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M81 41.4445C81 63.6303 63.0914 81.6154 41 81.6154C18.9086 81.6154 1 63.6303 1 41.4445C1 19.2587 18.9086 1.27356 41 1.27356C63.0914 1.27356 81 19.2587 81 41.4445Z" stroke="#65B9F4"/>
        <path d="M40 22.3633H41V59.5214H40V22.3633Z" stroke="#65B9F4"/>
        <path d="M22 41.4445V40.4402H59V41.4445H22Z" stroke="#65B9F4"/>
      </svg>
    </div>
    <div class="project-create-headline">
      <h3>Создать новый проект</h3>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CardCreateProject',
}
</script>

<style lang="scss" scoped>
.project{
  max-width: 300px;
  border: 1px dashed #65B9F4;
  border-radius: 4px;
  transition: background .3s ease-in-out;
  cursor: pointer;
  
  &-create{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;

    &-plus{
      height: 80px;
      width: 80px;
    }

    &-headline{
      margin-top: 20px;
      h3{
      font-family: "Open Sans";
      font-style: normal;
      font-weight: normal;
      font-size: 14px;
      color: #A7BED3;
      }
    }
  }

  &:hover{
    background: rgba(43, 82, 120, 0.05);
  }
}
</style>